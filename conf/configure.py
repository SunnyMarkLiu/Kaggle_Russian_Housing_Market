#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
全局配置文件
@author: MarkLiu
@time  : 17-5-23 上午10:33
"""
import time


class Configure(object):

    train_csv = '../data/train.csv'
    test_csv = '../data/test.csv'
    macro_csv = '../data/macro.csv'

    submission = '../result/submission_{}.csv'.format(time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time())))
