from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pandas as pd
import numpy as np

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

