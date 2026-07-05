import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

df=pd.read_csv('creditcard.csv')
df=df.dropna()
y=df['Class']
x = df.drop(['Time', 'Amount', 'Class'], axis=1)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

LR_Pipeline=Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

RF_Pipeline=Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

LR_Param={
    'classifier__C':[0.1, 1, 10],
    'classifier__solver':['lbfgs', 'liblinear']
}

RF_Param={
    'classifier__n_estimators':[100, 300],
    'classifier__max_depth':[5, None],
    'classifier__min_samples_split':[2,5]
}

LR_grid=GridSearchCV(
    LR_Pipeline,
    LR_Param,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)

RF_grid=GridSearchCV(
    RF_Pipeline,
    RF_Param,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)

LR_grid.fit(x_train, y_train)
RF_grid.fit(x_train, y_train)


print(LR_grid.best_params_)
print("LR Best ROC-AUC:", LR_grid.best_score_)
print(RF_grid.best_params_)
print("RF Best ROC-AUC:", RF_grid.best_score_)

LR_best=LR_grid.best_estimator_
RF_best=RF_grid.best_estimator_

y_pred1=LR_best.predict(x_test)
y_pred2=RF_best.predict(x_test)

y_prob1=LR_best.predict_proba(x_test)[:, 1]
y_prob2=RF_best.predict_proba(x_test)[:, 1]


print(classification_report(y_test, y_pred1))
print(classification_report(y_test, y_pred2))

fpr1, tpr1, _=roc_curve(y_test, y_prob1)
fpr2, tpr2, _=roc_curve(y_test, y_prob2)

auc1=roc_auc_score(y_test, y_prob1)
auc2=roc_auc_score(y_test, y_prob2)

plt.figure(figsize=(6, 6))
plt.plot(fpr1, tpr1, label=f"AUC = {auc1:.3f}")
plt.plot([0, 1], [0, 1], "--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.show()

plt.figure(figsize=(6, 6))
plt.plot(fpr2, tpr2, label=f"AUC = {auc2:.3f}")
plt.plot([0, 1], [0, 1], "--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest Classifier")
plt.legend()
plt.show()
