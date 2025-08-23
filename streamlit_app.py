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

import streamlit as st
import plotly.express as px

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="Economic Health Index",
    page_icon="📊",
    layout="wide"
)

# ----------------- Custom CSS -----------------
st.markdown("""
    <style>
        .main {background-color: #f8fafc;}
        h1, h2, h3 {color: #0f172a;}
        .metric-card {
            background: white;
            padding: 2rem;
            border-radius: 1rem;
            text-align: center;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- Title -----------------
st.title("📊 Economic Health Index")

# ----------------- Score Card -----------------
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h2>Latest Score</h2>
        <h1 style='color:#2563eb; font-size:3rem;'>{current_score}</h1>
        <p>Based on current macroeconomic data</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ----------------- Intro -----------------
with st.expander("ℹ️ Background", expanded=True):
    st.write(
        "Economic regimes are characterized by dynamic complexity and the movement of variables. "
        "Some movements coincide with economic downturns or rallies, while others show fluctuations during transitions. "
        "To separate economic regimes at their peak and trough, the choice of these variables is crucial for accurate classification."
    )

# ----------------- Indicators -----------------
tabs = st.tabs(["📈 Coincident", "⏳ Lagging", "🔮 Leading"])

with tabs[0]:
    st.write(
        "Coincident variables show significant fluctuations during economic downturns and expansions. "
        "Key indicators: Payroll employment, Civilian employment, Industrial production, and Unemployment rate."
    )

with tabs[1]:
    st.write(
        "Lagging indicators fluctuate after an economic downturn starts. "
        "Key indicators: Unemployment rate, CPI, and Bank interest rates."
    )

with tabs[2]:
    st.write(
        "Leading indicators include interest-rate spreads, credit/confidence gauges, and series that fall before a cycle peak. "
        "Examples: Treasury yield curve, broad money (M2), commodity prices, consumer sentiment, building permits."
    )

# ----------------- Problem & Solution -----------------
st.markdown("### ⚠️ Problem")
st.info(
    "The National Bureau of Economic Research (NBER) declares U.S. recessions months after they start, "
    "making real-time decision-making difficult."
)

st.markdown("### 💡 Solution")
st.success(
    "We use historical monthly macroeconomic indicators and NBER labels to train a logistic regression model. "
    "This model can estimate the probability of being in a recession and convert it into an **Economic Health Score (0–10)**."
)

# ----------------- Plot -----------------
st.markdown("### 📊 Economic Score Over Time")
st.plotly_chart(fig2, use_container_width=True)

st.plotly_chart(fig2, use_container_width=True)




