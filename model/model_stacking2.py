#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-26 上午9:30
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
# remove warnings
import warnings

warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.metrics import mean_squared_error
# my own module
from features import data_utils
from sklearn import preprocessing
from conf.configure import Configure

train, test, macro = data_utils.load_data()
print 'train:', train.shape
print 'test:', test.shape

# Deal with categorical values
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[c].values))
        train[c] = lbl.transform(list(train[c].values))

for c in test.columns:
    if test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(test[c].values))
        test[c] = lbl.transform(list(test[c].values))

cv_results = os.listdir('./cv_results')

df_train = train
df_test = test

for cv_result in cv_results:
    if 'train' in cv_result:
        df_train = pd.merge(df_train, pd.read_csv('./cv_results/' + cv_result), how='inner', on='id')
    if 'test' in cv_result:
        df_test = pd.merge(df_test, pd.read_csv('./cv_results/' + cv_result), how='inner', on='id')

print 'train:', df_train.shape
print 'test:', df_test.shape

y_train_all = df_train['price_doc']
id_train = df_train['id']
df_train.drop(['id', 'timestamp', 'price_doc'], axis=1, inplace=True)
id_test = df_test['id']
df_test.drop(['id', 'timestamp'], axis=1, inplace=True)

test_size = (1.0 * df_test.shape[0]) / df_train.shape[0]
print "submit test size:", test_size

# Convert to numpy values
X_all = df_train.values

# Create a validation set, with last 20% of data
num_train = int(df_train.shape[0] / (1 + test_size))

X_train_all = X_all

X_train = X_all[:num_train]
X_val = X_all[num_train:]
y_train = y_train_all[:num_train]
y_val = y_train_all[num_train:]
X_test = df_test
print "validate size:", 1.0 * X_val.shape[0] / X_train.shape[0]

df_columns = df_train.columns

print('X_train_all shape is', X_train_all.shape)
print('X_train shape is', X_train.shape)
print('y_train shape is', y_train.shape)
print('X_val shape is', X_val.shape)
print('y_val shape is', y_val.shape)
print('X_test shape is', X_test.shape)

dtrain_all = xgb.DMatrix(X_train_all, y_train_all, feature_names=df_columns)
dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
dval = xgb.DMatrix(X_val, y_val, feature_names=df_columns)
dtest = xgb.DMatrix(X_test, feature_names=df_columns)

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

num_round = 1000
xgb_params['nthread'] = 24
evallist = [(dval, 'eval')]

bst = xgb.train(xgb_params, dtrain, num_round, evallist, early_stopping_rounds=40, verbose_eval=10)

train_rmse = mean_squared_error(y_train, bst.predict(dtrain))
val_rmse = mean_squared_error(y_val, bst.predict(dval))
print 'train_rmse =', np.sqrt(train_rmse), ', val_rmse =', np.sqrt(val_rmse)

num_boost_round = bst.best_iteration
print 'best_iteration: ', num_boost_round
model = xgb.train(dict(xgb_params, silent=1), dtrain_all, num_boost_round=num_boost_round)

print 'predict submit...'
y_pred = model.predict(dtest)
df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
df_sub.to_csv(Configure.submission_path, index=False)
