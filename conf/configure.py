#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
全局配置文件
@author: MarkLiu
@time  : 17-5-23 上午10:33
"""
import time


class Configure(object):

    original_train_path = '../data/train.csv'
    original_test_path = '../data/test.csv'
    original_macro_path = '../data/macro.csv'

    original_imputed_train_path = '../data/imputed_train.csv'
    original_imputed_test_path = '../data/imputed_test.csv'
    original_imputed_macro_path = '../data/imputed_macro.csv'
    original_longitude_latitude_path = '../data/Subarea_Longitud_Latitud.csv'

    submission_path = '../result/submission_{}.csv'.format(time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time())))

    processed_train_path = '../data/processed_train_data.pkl'
    processed_test_path = '../data/processed_test_data.pkl'
    processed_macro_path = '../data/processed_macro_data.pkl'

    time_window_salecount_features_path = '../data/time_window_{}_subarea_salecount_features.pkl'

    multicollinearity_features = '../data/multicollinearity_features.pkl'
    time_window_salecount_features_path = '../data/time_window_{}_subarea_salecount_features.pkl'

    conbined_data_price_distance_path = '../data/conbined_data_price_distance_path.pkl'
