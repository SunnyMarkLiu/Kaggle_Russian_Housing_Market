#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-26 上午11:00
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
from conf.configure import Configure

xgboost_model_031051 = pd.read_csv('./my_best_result/xgboost_model_0.31051.csv')
xgboost_model_031191 = pd.read_csv('./my_best_result/xgboost_model_0.31191.csv')
xgboost_model_031211 = pd.read_csv('./my_best_result/xgboost_model_0.31211.csv')
xgboost_model_031284 = pd.read_csv('./my_best_result/xgboost_model_0.31284.csv')
xgboost_model_031289 = pd.read_csv('./my_best_result/xgboost_model_0.31289.csv')
xgboost_model_031295 = pd.read_csv('./my_best_result/xgboost_model_0.31295.csv')
xgboost_model_031323 = pd.read_csv('./my_best_result/xgboost_model_0.31323.csv')
xgboost_model_031328 = pd.read_csv('./my_best_result/xgboost_model_0.31328.csv')
xgboost_model_031344 = pd.read_csv('./my_best_result/xgboost_model_0.31344.csv')
xgboost_model_031392 = pd.read_csv('./my_best_result/xgboost_model_0.31392.csv')

average_result = 0.30 * (0.784 * xgboost_model_031051['price_doc'] + 0.216 * xgboost_model_031191['price_doc']) + \
                 0.25 * (0.784 * xgboost_model_031211['price_doc'] + 0.216 * xgboost_model_031284['price_doc']) + \
                 0.20 * (0.784 * xgboost_model_031289['price_doc'] + 0.216 * xgboost_model_031295['price_doc']) + \
                 0.15 * (0.784 * xgboost_model_031323['price_doc'] + 0.216 * xgboost_model_031328['price_doc']) + \
                 0.10 * (0.784 * xgboost_model_031344['price_doc'] + 0.216 * xgboost_model_031392['price_doc'])

df_sub = pd.DataFrame({'id': xgboost_model_031051['id'], 'price_doc': average_result})
df_sub.to_csv(Configure.submission_path, index=False)
