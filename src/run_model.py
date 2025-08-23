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

