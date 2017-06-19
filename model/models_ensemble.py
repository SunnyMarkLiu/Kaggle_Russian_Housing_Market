#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-18 下午6:20
"""
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
from conf.configure import Configure

model_1 = pd.read_csv('xgboost_model_1.csv') # 0.31191
model_2 = pd.read_csv('xgboost_model_2.csv') # 0.31284
model_3 = pd.read_csv('xgboost_model_3.csv') # 0.31499
model_4 = pd.read_csv('xgboost_model_4.csv') # 0.31039

# model ensemble
result = pd.DataFrame()
result['id'] = model_1['id']

result['price_doc'] = np.exp(0.20 * np.log(model_1['price_doc']) +
                             0.15 * np.log(model_2['price_doc']) +
                             0.10 * np.log(model_3['price_doc']) +
                             0.55 * np.log(model_4['price_doc']))

result.to_csv(Configure.submission_path, index=False)
