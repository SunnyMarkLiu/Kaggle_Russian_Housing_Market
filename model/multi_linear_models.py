#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-17 上午9:58
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
from sklearn import preprocessing
from sklearn.linear_model import ARDRegression, ElasticNet, HuberRegressor, Lars, \
    Lasso, LassoLars, LassoLarsIC, LinearRegression, PassiveAggressiveRegressor, \
    Ridge, SGDRegressor, TheilSenRegressor
from sklearn.svm import SVR

# my own module
from features import data_utils


def main():
    train, test, macro = data_utils.load_data()

    train['price_doc'] = np.log1p(train['price_doc'])
    ylog_train_all = train['price_doc']
    train.drop(['id', 'price_doc'], axis=1, inplace=True)
    submit_ids = test['id']
    test.drop(['id'], axis=1, inplace=True)

    # 合并训练集和测试集
    conbined_data = pd.concat([train[test.columns.values], test])
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

    test_size = (1.0 * test.shape[0]) / train.shape[0]
    print "submit test size:", test_size

    # Convert to numpy values
    X_all = train.values

    # Create a validation set, with last 20% of data
    num_train = int(train.shape[0] / (1+test_size))

    X_train_all = X_all
    X_train = X_all[:num_train]
    X_val = X_all[num_train:]
    ylog_train = ylog_train_all[:num_train]
    ylog_val = ylog_train_all[num_train:]
    X_test = test
    print "validate size:", 1.0*X_val.shape[0] / X_train.shape[0]

    print('X_train_all shape is', X_train_all.shape)
    print('X_train shape is', X_train.shape)
    print('y_train shape is', ylog_train.shape)
    print('X_val shape is', X_val.shape)
    print('y_val shape is', ylog_val.shape)
    print('X_test shape is', X_test.shape)

    print '=============== perform ARDRegression model ==============='
    ardr = ARDRegression(n_iter=300, verbose=True)
    ardr.fit(X_train, ylog_train)
    y_pred = np.exp(ardr.predict(X_test)) - 1
    df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': y_pred})
    df_sub.to_csv('../result/models/ARDRegression.csv', index=False)
    del ardr

    print '=============== perform ElasticNet model ==============='
    en = ElasticNet()
    en.fit(X_train, ylog_train)
    y_pred = np.exp(en.predict(X_test)) - 1
    df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': y_pred})
    df_sub.to_csv('../result/models/ElasticNet.csv', index=False)
    del en

    print '=============== perform HuberRegressor model ==============='
    hr = HuberRegressor()
    hr.fit(X_train, ylog_train)
    y_pred = np.exp(hr.predict(X_test)) - 1
    df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': y_pred})
    df_sub.to_csv('../result/models/HuberRegressor.csv', index=False)
    del hr

    print '=============== perform Lars model ==============='
    lars = Lars()
    lars.fit(X_train, ylog_train)
    y_pred = np.exp(lars.predict(X_test)) - 1
    df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': y_pred})
    df_sub.to_csv('../result/models/Lars.csv', index=False)
    del lars

    print '=============== perform Lasso model ==============='
    lasso = Lasso()
    lasso.fit(X_train, ylog_train)
    y_pred = np.exp(lasso.predict(X_test)) - 1
    df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': y_pred})
    df_sub.to_csv('../result/models/Lasso.csv', index=False)
    del lasso

    print '=============== perform LassoLars model ==============='
    lassolars = LassoLars()
    lassolars.fit(X_train, ylog_train)
    y_pred = np.exp(lassolars.predict(X_test)) - 1
    df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': y_pred})
    df_sub.to_csv('../result/models/LassoLars.csv', index=False)
    del lassolars

    print '=============== perform LassoLarsIC model ==============='
    lli = LassoLarsIC()
    lli.fit(X_train, ylog_train)
    y_pred = np.exp(lli.predict(X_test)) - 1
    df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': y_pred})
    df_sub.to_csv('../result/models/LassoLarsIC.csv', index=False)
    del lli

    print '=============== perform LinearRegression model ==============='
    lr = LinearRegression(n_jobs=-1)
    lr.fit(X_train, ylog_train)
    y_pred = np.exp(lr.predict(X_test)) - 1
    df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': y_pred})
    df_sub.to_csv('../result/models/LinearRegression.csv', index=False)
    del lr

    print '=============== perform PassiveAggressiveRegressor model ==============='
    par = PassiveAggressiveRegressor()
    par.fit(X_train, ylog_train)
    y_pred = np.exp(par.predict(X_test)) - 1
    df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': y_pred})
    df_sub.to_csv('../result/models/PassiveAggressiveRegressor.csv', index=False)
    del par

    print '=============== perform Ridge model ==============='
    ridge = Ridge()
    ridge.fit(X_train, ylog_train)
    y_pred = np.exp(ridge.predict(X_test)) - 1
    df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': y_pred})
    df_sub.to_csv('../result/models/Ridge.csv', index=False)
    del ridge

    print '=============== perform SGDRegressor model ==============='
    sgdr = SGDRegressor()
    sgdr.fit(X_train, ylog_train)
    y_pred = np.exp(sgdr.predict(X_test)) - 1
    df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': y_pred})
    df_sub.to_csv('../result/models/SGDRegressor.csv', index=False)
    del sgdr

    print '=============== perform TheilSenRegressor model ==============='
    tsr = TheilSenRegressor()
    tsr.fit(X_train, ylog_train)
    y_pred = np.exp(tsr.predict(X_test)) - 1
    df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': y_pred})
    df_sub.to_csv('../result/models/TheilSenRegressor.csv', index=False)
    del tsr

    print '=============== perform SVR rbf model ==============='
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1, verbose=20)
    svr_rbf.fit(X_train_all, ylog_train_all)
    y_pred = np.exp(svr_rbf.predict(X_test)) - 1
    df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': y_pred})
    df_sub.to_csv('../result/models/SVR_rbf.csv', index=False)
    del svr_rbf

    print '=============== perform SVR linear model ==============='
    svr_lin = SVR(kernel='linear', C=1e3, gamma=0.1, verbose=20)
    svr_lin.fit(X_train_all, ylog_train_all)
    y_pred = np.exp(svr_lin.predict(X_test)) - 1
    df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': y_pred})
    df_sub.to_csv('../result/models/SVR_linear.csv', index=False)
    del svr_lin

    print '=============== perform SVR poly model ==============='
    svr_poly = SVR(kernel='poly', C=1e3, degree=2, gamma=0.1,verbose=20)
    svr_poly.fit(X_train_all, ylog_train_all)
    y_pred = np.exp(svr_poly.predict(X_test)) - 1
    df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': y_pred})
    df_sub.to_csv('../result/models/SVR_poly.csv', index=False)
    del svr_poly

if __name__ == '__main__':
    main()
