import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
import pandera as pa

df = pd.read_csv('Dataset for Data Analytics.csv')

# ==================================== PHASE - 1 ====================================
print("Handling Missing Values...")

for features in df.columns:
    null_count = df[features].isnull().sum()
    if null_count == 0:
        continue
    
    missing_pct = (null_count / len(df)) * 100
    if (missing_pct <= 5):
        df.dropna(subset=[features], inplace=True)
    elif (missing_pct <= 20):
        if pd.api.types.is_numeric_dtype(df[features]):
            df[features] = df[features].fillna(df[features].median())
        else:
            # Finding Best correlated columns for Sub-Group Imputation
            df_clean = df.dropna()
            if df_clean.empty:
                df[features] = df[features].fillna(df[features].mode()[0] if not df[features].mode().empty else "Missing")
                continue
                
            x = df_clean.drop(columns=[features])
            original_cols = x.columns.tolist()
            x = pd.get_dummies(x, drop_first=True)
            y = df_clean[features]
            
            modelRFC1 = RandomForestClassifier(random_state=42)
            modelRFC1.fit(x, y)
            
            importances = pd.Series(modelRFC1.feature_importances_, index=x.columns)
            
            best_features = []
            for cols in importances.nlargest(5).index:
                for original_col in original_cols:
                    if (cols.startswith(original_col) and original_col not in best_features):
                        best_features.append(original_col)
                        
            top2_corr_cols = best_features[:2]
            
            if top2_corr_cols:
                mode_imputation = df.groupby(top2_corr_cols)[features].transform(lambda x: x.mode()[0] if not x.mode().empty else df[features].mode()[0])
                df[features] = df[features].fillna(mode_imputation)
            else:
                df[features] = df[features].fillna(df[features].mode()[0] if not df[features].mode().empty else "Missing")
            
    else:
        df_knn_buffer = df.copy()
        for col in df_knn_buffer.columns:
            if pd.api.types.is_numeric_dtype(df_knn_buffer[col]):
                df_knn_buffer[col] = df_knn_buffer[col].fillna(df_knn_buffer[col].median())
            else:
                if not df_knn_buffer[col].mode().empty:
                    col_mode = df_knn_buffer[col].mode()[0]
                else:
                    col_mode = "Missing"
                df_knn_buffer[col] = df_knn_buffer[col].fillna(col_mode)
                
                le = LabelEncoder()
                df_knn_buffer[col] = le.fit_transform(df_knn_buffer[col].astype(str))
                
        X_knn = df_knn_buffer.drop(columns=[features])
        
        if pd.api.types.is_numeric_dtype(df[features]):
            from sklearn.neighbors import KNeighborsRegressor
            modelKNN1 = KNeighborsRegressor(n_neighbors=7)
            
            # FIX: Force clean target values to floats, converting stray text strings to NaN safely
            y_clean = pd.to_numeric(df[features], errors='coerce')
            clean_idx = y_clean.notnull()
            
            modelKNN1.fit(X_knn[clean_idx], y_clean[clean_idx])
        else:
            from sklearn.neighbors import KNeighborsClassifier
            modelKNN1 = KNeighborsClassifier(n_neighbors=7)
            
            clean_idx = df[features].notnull()
            modelKNN1.fit(X_knn[clean_idx], df.loc[clean_idx, features])
        
        missing_idx = df[df[features].isnull()].index
        if not missing_idx.empty:
            predictions = modelKNN1.predict(X_knn.loc[missing_idx])
            df.loc[missing_idx, features] = predictions

print("Missing Values Handled Successfully")
print("-------------------------------------")

print("Handling Outliers...")

for features in df.columns:
    if pd.api.types.is_numeric_dtype(df[features]):      
        Q1 = df[features].quantile(0.25)
        Q3 = df[features].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[features] = df[features].clip(lower=lower_bound, upper=upper_bound)

print("Outliers Handled Successfully")
print("-------------------------------------")

#===================================== PHASE - 2 ====================================
print("Encoding Categorical Variables...")

non_numeric_df=df.select_dtypes(include=['object', 'string'])
count_labels=non_numeric_df.nunique()
low_cardinality = count_labels[count_labels <= 10].index.tolist()

if low_cardinality:
    df = pd.get_dummies(df, columns=low_cardinality, drop_first=True, dtype=int)

print("Categorical Variables Encoded Successfully")
print("-------------------------------------")

print("Fixing Multicollinearity...")

numeric_df=df.select_dtypes(include=[np.number])
corr_matrix=numeric_df.corr().abs()
upper_tri_matrix=corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_mask = (upper_tri_matrix > 0.8).any(axis=0)
high_corr_cols = high_corr_mask[high_corr_mask].index.tolist()
df=df.drop(columns=high_corr_cols)

print("Multicollinearity Fixed Successfully")
print("-------------------------------------")

#===================================== PHASE - 3 ====================================
print("Securing Database Design and Scheme using Pandera...")

inferred_schema=pa.infer_schema(df)
with open("schema.py", "w") as file:
    file.write(inferred_schema.to_script())
    
print("Database Design and Scheme Secured Successfully")
print("-------------------------------------")

df.to_csv("refined.csv", index=False)

