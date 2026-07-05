import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


full_df=pd.read_csv("BankChurners.csv")

# Scaling and PCA

cat_cols=full_df.select_dtypes(include='object')
df=full_df.drop(columns=cat_cols)
numeric_cols=df.columns

scaler=StandardScaler()
df=scaler.fit_transform(df)

pca=PCA(n_components=2)
df=pca.fit_transform(df)

# Elbow Method

inertias=[]
for i in range(1, 11):
    model=KMeans(n_clusters=i, random_state=42)
    model.fit(df)
    iner=model.inertia_
    inertias.append(iner)

plt.figure(figsize=(6,4))
sns.lineplot(x=range(1, 11), y=inertias, markers='o', linewidth=1.5)
plt.xlabel("No of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid()
plt.legend()
plt.show()

# Silhouette Method

scores=[]
for i in range(2, 11):
    model=KMeans(n_clusters=i, random_state=42)
    labels=model.fit_predict(df)
    score=silhouette_score(df, labels)
    scores.append(score)
    
plt.figure(figsize=(6,4))
sns.lineplot(x=range(2, 11), y=scores, palette='viridis')
plt.xlabel("No of Clusters")
plt.ylabel("Silhouette Score")
plt.legend()
plt.grid()
plt.show()

# Best Optimal KMeans Model (Based on Silhouette Scores)

max_val=max(scores)
best_k=scores.index(max_val) + 2

final_model=KMeans(n_clusters=best_k, random_state=42)
final_model.fit(df)

final_labels=final_model.labels_
final_centers=final_model.cluster_centers_

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df[:, 0],
    y=df[:, 1],
    hue=final_labels,
    palette='tab10'
)
sns.scatterplot(
    x=final_centers[:, 0],
    y=final_centers[:, 1],
    color='black',
    marker='X',
    s=250,
    label='Centroids'
)
plt.title("KMeans Segmentation")
plt.legend()
plt.show()

# Reverse Engineering the Centroids

scaled_ceters=pca.inverse_transform(final_centers)
original_centers=scaler.inverse_transform(scaled_ceters)

center_df=pd.DataFrame(original_centers, columns=numeric_cols)

print(center_df)