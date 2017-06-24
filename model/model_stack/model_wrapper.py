#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-24 下午1:30
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import xgboost as xgb
from abc import ABCMeta, abstractmethod


class BaseWrapper(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        return None


class SklearnWrapper(BaseWrapper):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x, y):
        self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict(x)


class XgbWrapper(BaseWrapper):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x, y):
        dtrain = xgb.DMatrix(x, label=y)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def cv_train(self, x, y, num_boost_round=1000, nfold=4, early_stopping_rounds=40):
        dtrain = xgb.DMatrix(x, label=y)
        res = xgb.cv(self.param, dtrain, num_boost_round=num_boost_round, nfold=nfold,
                     early_stopping_rounds=early_stopping_rounds, verbose_eval=10, show_stdv=True)

        best_nrounds = res.shape[0] - 1
        cv_mean = res.iloc[-1, 0]
        cv_std = res.iloc[-1, 1]
        return best_nrounds, cv_mean, cv_std

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))
