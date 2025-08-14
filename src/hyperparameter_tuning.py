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
