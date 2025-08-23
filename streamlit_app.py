import streamlit as st
from back_testing import temporal_mapping
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,precision_recall_curve
from sklearn.metrics import f1_score,make_scorer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import plotly.express as px
from src.preprocessing import feature_engineering,train_test_split
from src.hyperparameter_tuning import hyperparameter_tuning


data = pd.read_csv('data/Recession Indicators(Sheet1).csv',index_col='Date',
                   parse_dates=True).dropna()
X,y = feature_engineering(data)
X_train,y_train,X_test,y_test = train_test_split(X,y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
best_params = hyperparameter_tuning(X_train_scaled,y_train)
model = LogisticRegression(**best_params).fit(X_train_scaled,y_train)
X_test_scaled = scaler.transform(X_test)
logit_preds = model.predict(X_test_scaled)
logit_probs = model.predict_proba(X_test_scaled)
fig1, fig2 = temporal_mapping(logit_probs, y_test)
current_score = np.round((logit_probs[-1, 1] * 10),2)
previous_score = np.round((logit_probs[-2, 1] * 10),2)
delta_value = round(current_score - previous_score,2)
# ---------------- Streamlit UI ----------------
# Title & Description
st.title("📊 Economic Health Index")
# Latest Score
st.metric(
    label="Latest Score Based on Current Data",
    value=current_score,
    delta=delta_value,
    delta_color="normal"      
)

st.write("""
We use historical monthly macroeconomic indicators and **NBER recession labels** 
to train a Logistic Regression model. This model can:
- Estimate the probability of being in a recession this month.  
- Convert that probability into an **Economic Health Score (0–10 scale)**.  
""")
# Charts
st.subheader("📈 Economic Health Score Over Time")
st.plotly_chart(fig2, use_container_width=True)
st.subheader("🔍 Probability of Recession (Historical vs. Predicted)")
st.pyplot(fig1)


