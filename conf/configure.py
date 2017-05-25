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

    submission = '../result/submission_{}.csv'.format(time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time())))

    processed_train_path = '../data/processed_train_data.pkl'
    processed_test_path = '../data/processed_test_data.pkl'
    processed_macro_path = '../data/processed_macro_data.pkl'
