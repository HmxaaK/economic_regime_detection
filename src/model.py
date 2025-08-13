"""
Economic Regime Nowcasting Model
--------------------------------
Author: Hamza
Description:
Logistic Regression model to estimate the probability of recession in real time
and convert it into an economic health score, based on macroeconomic indicators.
"""

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
def feature_engineering(data_set):
  #specifying the target and features
  target = 'USREC'
  lvl_features = data_set.drop(columns=[target])
  #Taking polynomial of features
  poly = PolynomialFeatures(degree=2,include_bias=False)
  poly_arr = poly.fit_transform(lvl_features)
  poly_names = poly.get_feature_names_out(input_features=lvl_features.columns)
  poly_features = pd.DataFrame(poly_arr, index=lvl_features.index,
                               columns=poly_names)
  #Taking log,log_difference and difference
  log_names = []
  diff_names = []
  for var in lvl_features.columns:
    if (lvl_features[var] <= 0).any():
      diff_names.append(var)
    else:
      log_names.append(var)
  log_features = np.log(lvl_features[log_names])
  log_features.columns = ['Log' + item for item in log_names]
  log_diff = log_features - log_features.shift()
  log_diff.columns = ['Logdiff' + item for item in log_names]
  #Final Features set
  X = pd.concat([poly_features,log_features,log_diff],axis=1).dropna()
  y = data[target]
  y = y.loc[X.index]
  return X,y


def train_test_split(features,target):
  train_size = int(len(features)*0.60)
  X_train,y_train=features.iloc[:train_size,:],target.iloc[:train_size]
  X_test,y_test=features.iloc[train_size:,:],target.iloc[train_size:]
  return X_train,y_train,X_test,y_test

def hyperparameter_tuning(features,target):
  model = LogisticRegression()
  arr = np.logspace(-6,3,100)
  param_grid = {'C': arr,
                'penalty': ['l1','l2'],
                'solver' : ['liblinear'],
                'class_weight':[None, 'balanced'],
                'random_state':[42],
                'max_iter': [1000],
                 }
  tsvc = TimeSeriesSplit(n_splits=3)
  f1_scorer = make_scorer(f1_score)
  grid_search = GridSearchCV(estimator=model,param_grid=param_grid,
                             cv=tsvc,scoring=f1_scorer)
  grid_search.fit(features,target)
  best_params = grid_search.best_params_
  return best_params

def model_performance(y_actual,y_predict):
  recall = recall_score(y_actual,y_predict)
  precision = precision_score(y_actual,y_predict)
  f1 = f1_score(y_actual,y_predict)
  cm = confusion_matrix(y_actual,y_predict)
  disp_cm = ConfusionMatrixDisplay(cm)
  print (f'Recall : {recall:.2f}')
  print (f'Precision : {precision:.2f}')
  print (f'F1 : {f1:.2f}')
  disp_cm.plot()
  plt.show()



data = pd.read_csv('Recession Indicators(Sheet1).csv',index_col='Date',
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

def temporal_mapping(probs, labels):
    probs_recession = probs[:,1] 
    plt.figure(figsize=(12, 6)) 
    plt.plot(labels.index, probs_recession, label="Probability of Recession")
    mask = (labels == 1).to_numpy()
    plt.fill_between(
        labels.index, 0, probs_recession,
        where=mask,
        color='grey', label="NBER = Recession"
    )
    plt.ylim(0,1)
    plt.ylabel("Probability")
    plt.legend()
    plt.show()
    
    economic_health_scores = np.round((probs[-39:, 0] * 10),2)
    fig = px.line(x=labels.index[-39:],y=economic_health_scores,
                  labels={"x":"","y":"Economic Health Score"})
    fig.show()


temporal_mapping(logit_probs, y_test)


