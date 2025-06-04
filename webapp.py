import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Telco.csv")

df.drop(['customerID'], axis=1, inplace=True)
print(df)
print(df.isnull().sum())
df.columns

df.info()
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

X = df.drop("Churn", axis=1)
y = df["Churn"]

df.info()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

from xgboost import plot_importance
plot_importance(model)
plt.show()
sns.countplot(data=df, x='Churn')
plt.title('Target')
plt.show()

df.corr(numeric_only=True)['Churn'].sort_values(ascending=False).plot(kind='bar')
plt.title('Churn')
plt.show()

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy:.4f}")

from sklearn.metrics import confusion_matrix, classification_report
print("혼동 행렬 : \n",confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid = GridSearchCV(xgb.XGBClassifier(eval_metric='logloss'),
                    param_grid, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)

print("최적 파라미터:", grid.best_params_)

from xgboost import XGBClassifier

# 최적 파라미터를 사용해 모델 생성
model = XGBClassifier(**grid.best_params_)

# 모델 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy:.4f}")