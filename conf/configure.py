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
    original_BAD_ADDRESS_FIX_path = '../data/BAD_ADDRESS_FIX.xlsx'

    original_imputed_train_path = '../data/imputed_train.csv'
    original_imputed_test_path = '../data/imputed_test.csv'
    original_imputed_macro_path = '../data/imputed_macro.csv'
    original_longitude_latitude_path = '../data/Subarea_Longitud_Latitud.csv'

    submission_path = '../result/submission_{}.csv'.format(time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time())))

    processed_train_path = '../data/processed_train_data.pkl'
    processed_test_path = '../data/processed_test_data.pkl'
    processed_macro_path = '../data/processed_macro_data.pkl'

    groupby_time_window_salecount_features_path = '../data/time_window_{}_subarea_salecount_features_pre_{}_days.pkl'
    single_time_window_salecount_features_path = '../data/time_window_{}_salecount_features_pre_{}_days.pkl'

    multicollinearity_features = '../data/multicollinearity_features.pkl'

    conbined_data_price_distance_path = '../data/conbined_data_price_distance_path.pkl'
    train_cv_result_for_model_stacking = '../model/cv_results/{}_train__cv_result_for_model_stacking_{}.csv'
    test_cv_result_for_model_stacking = '../model/cv_results/{}_test__cv_result_for_model_stacking_{}.csv'
