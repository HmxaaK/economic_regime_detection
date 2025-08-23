import streamlit as st
st.title('Economic Health Index')
st.subheader('Macroeconomic Indicators for Regime Classification')
st.write('Economic regimes are characterized by dynamic complexity and the movement of variables. Some movements coincide with the economic downturns or rallies, while others show fluctuations during transitions. In order to separate economic regimes at their peak and trough, the choice of these variables is crucial for accurate classification') 
st.subheader('Coincident Indicators')
st.write('The variables that show significant fluctuations during the economic downturns and expansion are called coincident economic variables. The most significant of these in identifying economic regimes are Payroll employment, Civilian employment, Industrial production, and Unemployment rate.')
st.subheader('Lagging Indicators')
st.write('These indicators fluctuate after an economic downturn starts. The unemployment rate is the most significant indicator in this context, as it spikes greatly after the start of the recession and falls when the economic activity reaches its peak. Significant lagging variables include CPI and Bank interest rates, as indicated by (Hiroshi Iyetomi, 2020).')
st.subheader('Leading Indicators')
st.write('Leading indicators include interest-rate spreads, credit and confidence gauges, and other series that typically fall before a cycle peak. For example, the Treasury yield curve (e.g., 10-year minus 2-year or 3-month yields) has inverted before every post-1970 U.S. recession. (Luca Benzoni, 2018). Likewise, broad money (M2) and commodity price indexes often reach their peak before an economic downturn. (Michael D. Bordo, 2004 ). Consumer sentiment also tends to decline before the onset of an economic recession. (Hiroshi Iyetomi, 2020). The Conference Board’s Leading Economic Index even explicitly includes building permits and yield spreads as components.') 
st.subheader('Problem')
st.write('The National Bureau of Economic Research (NBER) declares U.S. recessions months after they start. This lag makes real-time decision-making difficult for policymakers, investors, and businesses.')
st.subheader('Solution')
st.write('We use historical monthly macroeconomic indicators and NBER labels to train a logistic regression model. This model can:')
st.write('-Estimate the probability of being in a recession this month.')
st.write('-Convert that probability into an Economic Health Score (0–10 scale).')
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
from preprocessing import feature_engineering,train_test_split
from hyperparameter_tuning import hyperparameter_tuning
from evaluation import model_performance,temporal_mapping

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
model_performance(y_test,logit_preds)
temporal_mapping(logit_probs, y_test)
