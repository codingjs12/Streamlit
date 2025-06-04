import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
from xgboost import plot_importance, XGBClassifier

st.title("💼 Telco 고객 이탈 예측")

# 📥 GitHub에서 직접 데이터 로딩
CSV_URL = "https://raw.githubusercontent.com/codingjs12/streamlit/main/Telco.csv"  # ← 여기를 본인 주소로 수정

@st.cache_data
def load_data(url):
    return pd.read_csv(url)

df = load_data(CSV_URL)

# 데이터 전처리
if 'customerID' in df.columns:
    df.drop(['customerID'], axis=1, inplace=True)

st.subheader("📊 원본 데이터")
st.dataframe(df.head())

st.write("결측치 수:")
st.write(df.isnull().sum())

    # 인코딩
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

X = df.drop("Churn", axis=1)
y = df["Churn"]

    # 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 모델 학습
model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

    # 중요도 시각화
st.subheader("📈 Feature 중요도")
fig1, ax1 = plt.subplots()
plot_importance(model, ax=ax1)
st.pyplot(fig1)

    # Churn 분포 시각화
st.subheader("📌 Churn 분포")
fig2, ax2 = plt.subplots()
sns.countplot(data=df, x='Churn', ax=ax2)
st.pyplot(fig2)

    # 상관관계 시각화
st.subheader("🔗 Churn 상관관계")
fig3, ax3 = plt.subplots()
df.corr(numeric_only=True)['Churn'].sort_values(ascending=False).plot(kind='bar', ax=ax3)
plt.title('Churn 상관관계')
st.pyplot(fig3)

    # 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.subheader("🎯 기본 모델 정확도")
st.write(f"정확도: {accuracy:.4f}")

st.subheader("📌 혼동 행렬")
st.text(confusion_matrix(y_test, y_pred))

st.subheader("📌 분류 리포트")
st.text(classification_report(y_test, y_pred))

    # 하이퍼파라미터 튜닝
with st.expander("⚙️ GridSearchCV로 튜닝하기"):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    }

    grid = GridSearchCV(xgb.XGBClassifier(eval_metric='logloss'),
                            param_grid, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)

    st.write("최적 파라미터:")
    st.write(grid.best_params_)

        # 최적 모델로 재학습 및 평가
    tuned_model = XGBClassifier(**grid.best_params_)
    tuned_model.fit(X_train, y_train)
    tuned_pred = tuned_model.predict(X_test)
    tuned_acc = accuracy_score(y_test, tuned_pred)

    st.subheader("✅ 튜닝 후 정확도")
    st.write(f"정확도: {tuned_acc:.4f}")