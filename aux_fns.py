import numpy as np
import pandas as pd
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import time
import os
import json
import multiprocessing
import csv

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def load_res(p, f):
    with open(str(p)+str(f), 'r') as JSON:
        json_dict = json.load(JSON)
    return np.asarray(json.loads(json_dict))

def my_cross_val_predict(X, Y, my_kfold, estimators):
    x_test_all = np.ndarray([0, X.shape[1]])
    y_test_all = np.ndarray([0])
    y_test_pred_all = np.ndarray([0])
    indices_all = np.ndarray([0])
    feats_all = []

    y_test_all_ = dict()
    y_test_pred_all_ = dict()
    indices_all_ = dict()

    for i, (train_index, test_index) in enumerate(my_kfold.split(X, Y)):
        x = X[train_index, :]
        y = Y[train_index]
        x_test = X[test_index, :]
        y_test = Y[test_index]

        M = estimators.fit(x, y)

        y_test_pred = M.predict(x_test)
        M_feats = M.best_estimator_.named_steps['mod'].feature_importances_
        feats_all.append(M_feats)
        x_test_all = np.concatenate([x_test_all, x_test], axis=0)
        y_test_all = np.concatenate([y_test_all, y_test], axis=0)
        y_test_pred_all = np.concatenate([y_test_pred_all, y_test_pred], axis=0)
        indices_all = np.concatenate([indices_all, test_index], axis=0)
        
        y_test_all_.update({'fold'+str(i): y_test})
        y_test_pred_all_.update({'fold'+str(i): y_test_pred})
        indices_all_.update({'fold'+str(i): test_index})
    
    return x_test_all, y_test_all, y_test_pred_all, indices_all, feats_all, y_test_all_, y_test_pred_all_, indices_all_, feats_all

def ML_nested(X, Y, prefix='ms'):
    
    # Set up possible values of parameters to optimize over
    param_dist = {
        'mod__n_estimators': range(150, 200, 5),
        'mod__max_depth': range(2, 15, 2),
        'mod__learning_rate': np.linspace(0.01, 0.05, 11),
        'mod__subsample': np.linspace(0.7, 0.9, 21),
        'mod__colsample_bytree': np.linspace(0.48, 0.98, 11),
        'mod__min_child_weight': range(1, 9, 1),
    }

    # gradient boosted regression; Pipe ensures scaling based on training data only
    mod_pipe = Pipeline([('scaler', StandardScaler()),('mod', xgb.XGBRegressor(n_jobs=1))])

    # Arrays to store scores
    nested_y = np.empty([Y.shape[1], Y.shape[0]])
    nested_preds = np.empty([Y.shape[1], Y.shape[0]])
    feats = []

    nested_y_ = dict()
    nested_preds_ = dict()
    nested_indices_ = dict()
    
    # Loop for each trial
    for ix in range(Y.shape[1]):
        print('timepoint: ', ix)
        mask = np.isnan(Y[:, ix])
        X_ = X[~mask]  # remove missing values - important for predicting measured responses
        Y_ = Y[~mask, ix]
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=ix)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=ix)

        # Non_nested parameter search and scoring
        grid = RandomizedSearchCV(mod_pipe, param_dist, cv=inner_cv,
                                  scoring='neg_mean_squared_error',
                                  n_iter=1000, n_jobs=-2, verbose=1)

        x_test_all, y_test_all, y_test_pred_all, indices_all, feats_all, y_test_all_, y_test_pred_all_, indices_all_, feats_all = my_cross_val_predict(X_.values, Y_, outer_cv, grid)
        feats.append(feats_all)        
        
        nested_y[ix, ~mask] = y_test_all
        nested_preds[ix, ~mask] = y_test_pred_all
        nested_y_.update({'tp '+str(ix): y_test_all_})
        nested_preds_.update({'tp '+str(ix): y_test_pred_all_})
        nested_indices_.update({'tp '+str(ix): indices_all_})

    json_dump = json.dumps(nested_preds, cls=NumpyEncoder)
    with open(prefix+'_preds', 'w') as f:
        json.dump(json_dump, f)
    
    json_dump = json.dumps(nested_y, cls=NumpyEncoder)
    with open(prefix+'_y', 'w') as f:
        json.dump(json_dump, f)

    json_dump = json.dumps(nested_preds_, cls=NumpyEncoder)
    with open(prefix+'_preds_dict', 'w') as f:
        json.dump(json_dump, f)
    
    json_dump = json.dumps(nested_y_, cls=NumpyEncoder)
    with open(prefix+'_y_dict', 'w') as f:
        json.dump(json_dump, f)

    json_dump = json.dumps(nested_indices_, cls=NumpyEncoder)
    with open(prefix+'_indices_dict', 'w') as f:
        json.dump(json_dump, f)

    features = pd.DataFrame(feats_all)
    features.to_csv(prefix+"_features.csv")

    return
    
