import numpy as np
import pandas as pd
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import time
import os
import json
import multiprocessing
import csv
from aux_fns import *

# # load data
# df = pd.read_csv('DMS_2na.csv', header=0, index_col=0) # Data with OGTT and predictor features
# rs_derived = pd.read_csv('rs_derived.csv', header=None) # residuals in personalized eDES
  
# # median response simulations
# med_sim_glu = pd.read_csv('DMS_median_glu.csv', index_col=0)
# med_sim_ins = pd.read_csv('DMS_median_ins.csv', index_col=0)
# res_tohealthy = df.iloc[:, 7:21].values - np.hstack((med_sim_glu.iloc[0, [0, 15, 30, 45, 60, 90, 120]],
#                                                      med_sim_ins.iloc[0, [0, 15, 30, 45, 60, 90, 120]]))

# OGTT14 = df.iloc[:, 7:21]

# # preprocess for model building
# df = df.drop(['Glucose_t0_FP', 'Glucose_t120_FP', 'N_CP0_30_G0_30_ratio',
#               'N_GTS_WHO', 'ID', 'N_HOMA2_IR', 'N_all_Matsuda_IR', 'HIRI_7'], axis=1)  # remove OGTT derived features from predictors

# categoricals = df.iloc[:, 22:-1].dtypes == 'object'
# cats = categoricals[categoricals == True].index.tolist()
# predictors_ = df.iloc[:, 22:-1]  # deselect glucose, insulin and cpep time series
# predictors = pd.get_dummies(predictors_, columns=cats, drop_first=True)  # one-hot encoding of categorical variables
# X = predictors

# # targets
# Y1 = rs_derived.iloc[:, [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]].values
# Y2 = res_tohealthy
# Y3 = OGTT14.values

# # train and evaluate XGBoost models
# ML_nested(X, Y1, prefix='perso_') # Hybrid II.
# ML_nested(X, Y2, prefix='avg_healthy') # Hybrid I.
# ML_nested(X, Y3, prefix='pure_ml') # Ref. GBR

# load generated data:
X = pd.DataFrame(np.random.rand(10,5)) # random matrix of predictors with 10 individuals and 5 features
Y = pd.DataFrame(np.random.rand(10,1)) # random vector of targets
ML_nested(X, Y, prefix='TEST') # test case with random predictors and target

# process saved results

path = ''
files = ['TEST_preds_dict','TEST_indices_dict']
names = ['pred', 'idxs']

d = {}
for ix_, f in enumerate(files):
    d[names[ix_]] = load_res(path, f)
    
r2 = np.empty((5,1))
mse = np.empty((5,1))
preds = np.empty((20,1))

for i, (key, value) in enumerate(d['idxs'].item().items()):
    nan_mask = ~np.isnan(Y[:,i])
    preds_ = np.empty(nan_mask[nan_mask == True].shape)
    print(preds_.shape)
    for j, (key_, value_) in enumerate(d['idxs'].item()[key].items()):
        y = Y[nan_mask, i] # OGTT.values[~np.isnan(OGTT.values[:,i]), i]
        y = y[d['idxs'].item()[key][key_]]
        
        #  sim_ = sim[~nan_mask, i]
        #  sim_ = sim_[d['idxs'].item()[key][key_]]
        
        y_p = d['pred'].item()[key][key_] # + sim_
        r2[j,i] = r2_score(y, y_p) 
        mse[j,i] = mean_squared_error(y, y_p)
        indices = np.asarray(d['idxs'].item()[key][key_])

        ppreds_[indices] = y_p
        
    preds[nan_mask, i] = preds_
