#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
处理 multicollinearity 问题
REF： https://www.kaggle.com/ffisegydd/sklearn-multicollinearity-class

@author: MarkLiu
@time  : 17-6-6 上午10:30
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer

from statsmodels.stats.outliers_influence import variance_inflation_factor

# remove warnings
import warnings

warnings.filterwarnings('ignore')

# my own module
import data_utils


class ReduceVIF(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh

        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            self.imputer = Imputer(strategy=impute_strategy)

        self.drop_features = []

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return self.calculate_vif(X, self.thresh)

    def calculate_vif(self, X, thresh=5.0):
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped = True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print('Dropping {} with vif={}'.format(X.columns[maxloc], max_vif))
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped = True
                self.drop_features.append(X.columns.tolist()[maxloc])
        return X


def deal_multicollinearity(train_num, test_num, y):
    transformer = ReduceVIF(impute=False)
    train_num = transformer.fit_transform(train_num, y)
    test_num.drop([transformer.drop_features], axis=1, inplace=True)
    return train_num, test_num


def main():
    print 'loading train and test datas...'
    train, test, _ = data_utils.load_data()
    print 'train:', train.shape, ', test:', test.shape

    train_id = train['id']
    train_price_doc = train['price_doc']
    train.drop(['id', 'price_doc'], axis=1, inplace=True)
    test_id = test['id']
    test.drop(['id'], axis=1, inplace=True)

    str_columns = train.select_dtypes(include=['object']).columns.values.tolist()
    num_columns = train.select_dtypes(exclude=['object']).columns.values.tolist()

    num_columns.remove('timestamp')

    train_num, test_num = deal_multicollinearity(train[num_columns], test[num_columns], train_price_doc)

    train = pd.concat([train_num, train[str_columns]], axis=1)
    test = pd.concat([test_num, test[str_columns]], axis=1)

    train['id'] = train_id
    train['price_doc'] = train_price_doc
    test['id'] = test_id
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== dealing with multicollinearity problem =============="
    main()
