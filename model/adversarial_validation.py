#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-7 下午8:17
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
import xgboost as xgb
# remove warnings
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class AdversarialValidation(object):
    """ AdversarialValidation 
    http://fastml.com/adversarial-validation-part-two/ 
    
    1. Train a classifier to identify whether data comes from the train or test set.
    2. Sort the training data by it’s probability of being in the test set.
    3. Select the training data most similar to the test data as your validation set.
    """

    def __init__(self, train, test, xgb_params=None, error_predict_prob=0.4):
        self.train = train
        self.test = test
        if xgb_params is None:
            xgb_params = {
                'learning_rate': 0.05, 'max_depth': 4, 'subsample': 0.9,
                'colsample_bytree': 0.9, 'objective': 'binary:logistic',
                'silent': 1, 'n_estimators': 100, 'gamma': 1,
                'min_child_weight': 4,
                'seed': 100
            }
        self.xgb_params = xgb_params
        self.error_predict_prob = error_predict_prob

    def generate(self):
        """ 生成训练集和特征分布与测试集特征分布相一致的验证集 """
        pre_train = self.train.copy()
        pre_test = self.test.copy()

        self.train['price_doc'] = np.log1p(self.train['price_doc'])
        train_ids = self.train['id']
        ylog_train_all = self.train['price_doc']
        self.train.drop(['id', 'price_doc'], axis=1, inplace=True)
        submit_ids = self.test['id']
        self.test.drop(['id'], axis=1, inplace=True)

        self.train['istest'] = 0
        self.test['istest'] = 1
        x = pd.concat([self.train, self.test])
        y = x['istest']
        x.drop(['istest'], axis=1, inplace=True)

        train_y = self.train['istest']
        train_x = self.train.drop(['istest'], axis=1)

        clf = xgb.XGBClassifier(**self.xgb_params)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)

        for train_index, test_index in skf.split(x, y):
            x0, x1 = x.iloc[train_index], x.iloc[test_index]
            y0, y1 = y.iloc[train_index], y.iloc[test_index]
            clf.fit(x0, y0, eval_set=[(x1, y1)],
                    eval_metric='logloss', verbose=False, early_stopping_rounds=10)

            prval = clf.predict_proba(x1)[:, 1]
            print 'predict auc:', roc_auc_score(y1, prval)

        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.4, random_state=0)
        dtrain_all = xgb.DMatrix(x, y, feature_names=x.columns)
        dtrain = xgb.DMatrix(X_train, y_train, feature_names=x.columns)
        dval = xgb.DMatrix(X_val, y_val, feature_names=x.columns)
        plst = self.xgb_params.items()
        plst += [('eval_metric', 'auc')]
        evallist = [(dval, 'eval')]
        num_round = 100
        bst = xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=100, verbose_eval=10)
        model = xgb.train(dict(self.xgb_params, silent=1), dtrain_all, num_boost_round=bst.best_iteration)
        dtrain = xgb.DMatrix(train_x, train_y, feature_names=x.columns)
        predict_istest_prob = model.predict(dtrain)
        predict_istest_prob = pd.DataFrame({'id': train_ids,
                                            'istest_prob': predict_istest_prob})
        predict_istest_prob = predict_istest_prob.sort_values(by='istest_prob', ascending=False)
        validate_ids = predict_istest_prob[predict_istest_prob['istest_prob'] > self.error_predict_prob].index.tolist()
        train_ids = set(pre_train.index.copy().tolist())
        train_ids = train_ids.difference(validate_ids)
        validate = pre_train.loc[validate_ids]
        train = pre_train.loc[train_ids]
        self.test.drop(['istest'], axis=1, inplace=True)
        test = self.test

        return train, test, validate
