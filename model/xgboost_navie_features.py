#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
使用基本的特征构建基本的 xgboost 模型
@author: MarkLiu
@time  : 17-5-25 下午9:03
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
from sklearn import preprocessing

# my own module
from features import data_utils
from conf.configure import Configure
from adversarial_validation import AdversarialValidation


def main():
    train, test, macro = data_utils.load_data()

    train_ids = train['id']
    train['price_doc'] = np.log1p(train['price_doc'])
    y_train_all = train['price_doc']
    train.drop(['id', 'price_doc'], axis=1, inplace=True)
    submit_ids = test['id']
    test.drop(['id'], axis=1, inplace=True)

    # 合并训练集和测试集
    conbined_data = pd.concat([train[test.columns.values], test])
    # macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
    #               "micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
    #               "income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build", "timestamp"]
    # conbined_data = pd.merge_ordered(conbined_data, macro[macro_cols], on='timestamp', how='left')

    conbined_data.drop(['timestamp'], axis=1, inplace=True)
    print "conbined_data:", conbined_data.shape

    # Deal with categorical values
    for c in conbined_data.columns:
        if conbined_data[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(conbined_data[c].values))
            conbined_data[c] = lbl.transform(list(conbined_data[c].values))

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    test_size = (1.0 * test.shape[0]) / (train.shape[0] + test.shape[0])
    print "submit test size:", test_size

    dtrain_all = xgb.DMatrix(train.values, y_train_all, feature_names=train.columns)

    train['price_doc'] = y_train_all
    train['id'] = train_ids
    test['id'] = submit_ids

    # 应用adversarial_validation 生成验证集
    train, test, validate = AdversarialValidation(train, test).generate()

    y_train = train['price_doc']
    y_validate = validate['price_doc']

    train.drop(['id', 'price_doc'], axis=1, inplace=True)
    validate.drop(['id', 'price_doc'], axis=1, inplace=True)

    print('train shape is', train.shape)
    print('y_train shape is', y_train.shape)
    print('validate shape is', validate.shape)
    print('y_validate shape is', y_validate.shape)
    print('test shape is', test.shape)

    dtrain = xgb.DMatrix(train.values, y_train, feature_names=train.columns)
    dval = xgb.DMatrix(validate.values, y_validate, feature_names=train.columns)
    dtest = xgb.DMatrix(test.values, feature_names=train.columns)

    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'silent': 1
    }

    num_round = 1000
    xgb_params['nthread'] = 24
    plst = xgb_params.items()
    plst += [('eval_metric', 'rmse')]
    evallist = [(dval, 'eval')]

    bst = xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=100, verbose_eval=10)

    num_boost_round = bst.best_iteration
    print 'best_iteration: ', num_boost_round
    # model = xgb.train(dict(xgb_params, silent=1), dtrain_all, num_boost_round=num_boost_round)

    print 'predict submit...'
    ylog_pred = bst.predict(dtest)
    y_pred = np.exp(ylog_pred) - 1
    df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': y_pred})
    df_sub.to_csv(Configure.submission_path, index=False)


if __name__ == '__main__':
    main()
