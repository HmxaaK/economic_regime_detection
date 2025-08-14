def train_test_split(features,target):
  train_size = int(len(features)*0.60)
  X_train,y_train=features.iloc[:train_size,:],target.iloc[:train_size]
  X_test,y_test=features.iloc[train_size:,:],target.iloc[train_size:]
  return X_train,y_train,X_test,y_test
    
