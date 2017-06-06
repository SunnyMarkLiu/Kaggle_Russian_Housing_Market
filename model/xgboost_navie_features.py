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
# my own module
from features import data_utils
from conf.configure import Configure


def main():
    train, test, macro = data_utils.load_data()

    train['price_doc'] = np.log1p(train['price_doc'])
    ylog_train_all = train['price_doc']
    train.drop(['id', 'price_doc'], axis=1, inplace=True)
    submit_ids = test['id']
    test.drop(['id'], axis=1, inplace=True)

    # 合并训练集和测试集
    conbined_data = pd.concat([train[test.columns.values], test])
    macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
                  "micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
                  "income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build", "timestamp"]
    conbined_data = pd.merge_ordered(conbined_data, macro[macro_cols], on='timestamp', how='left')
    # conbined_data.columns = test.columns.values
    conbined_data.drop(['timestamp'], axis=1, inplace=True)
    print "conbined_data:", conbined_data.shape

    # str_columns = conbined_data.select_dtypes(include=['object']).columns.values.tolist()

    # Deal with categorical values
    df_numeric = conbined_data.select_dtypes(exclude=['object'])
    df_obj = conbined_data.select_dtypes(include=['object']).copy()

    for c in df_obj:
        df_obj[c] = pd.factorize(df_obj[c])[0]

    conbined_data = pd.concat([df_numeric, df_obj], axis=1)

    # dummy code
    # dummies_data = pd.get_dummies(conbined_data[str_columns])
    # conbined_data[dummies_data.columns] = dummies_data[dummies_data.columns]
    # conbined_data.drop(str_columns, axis=1, inplace=True)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    test_size = (1.0 * test.shape[0]) / (train.shape[0] + test.shape[0])
    print "submit test size:", test_size

    # Convert to numpy values
    X_all = train.values
    print(X_all.shape)

    # Create a validation set, with last 20% of data
    num_train = train.shape[0]
    num_val = int(num_train * 0.2)

    X_train_all = X_all[:num_train]
    X_train = X_all[:num_train - num_val]
    X_val = X_all[num_train - num_val:num_train]
    ylog_train = ylog_train_all[:-num_val]
    ylog_val = ylog_train_all[-num_val:]

    X_test = test

    df_columns = train.columns

    print('X_train_all shape is', X_train_all.shape)
    print('X_train shape is', X_train.shape)
    print('y_train shape is', ylog_train.shape)
    print('X_val shape is', X_val.shape)
    print('y_val shape is', ylog_val.shape)
    print('X_test shape is', X_test.shape)

    dtrain_all = xgb.DMatrix(X_train_all, ylog_train_all, feature_names=df_columns)
    dtrain = xgb.DMatrix(X_train, ylog_train, feature_names=df_columns)
    dval = xgb.DMatrix(X_val, ylog_val, feature_names=df_columns)
    dtest = xgb.DMatrix(X_test, feature_names=df_columns)

    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 1.0,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    num_round = 1000
    xgb_params['nthread'] = 24
    # param['eval_metric'] = "auc"
    plst = xgb_params.items()
    plst += [('eval_metric', 'rmse')]
    evallist = [(dval, 'eval')]

    bst = xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=100, verbose_eval=10)

    num_boost_round = bst.best_iteration
    print 'best_iteration: ', num_boost_round
    model = xgb.train(dict(xgb_params, silent=1), dtrain_all, num_boost_round=num_boost_round)

    print 'predict submit...'
    ylog_pred = model.predict(dtest)
    y_pred = np.exp(ylog_pred) - 1
    df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': y_pred})
    df_sub.to_csv(Configure.submission_path, index=False)


if __name__ == '__main__':
    main()
