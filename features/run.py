#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-5-23 上午11:57
"""
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

cmd = 'rm ../data/processed_*.pkl'
os.system(cmd)

cmd = 'python impute_missing_data.py'
os.system(cmd)

cmd = 'python subsample_traindata.py'
os.system(cmd)

cmd = 'python train_test_preprocess.py'
os.system(cmd)

cmd = 'python generate_neighbourhood_features.py'
os.system(cmd)

cmd = 'python generate_longitude_latitude_features.py'
os.system(cmd)
#
# cmd = 'python ratio_dispersed_features.py'
# os.system(cmd)

# cmd = 'python deal_multicollinearity.py'
# os.system(cmd)

# cmd = 'python generate_time_window_features.py'
# os.system(cmd)

# cmd = 'python subsample_traindata.py'
# os.system(cmd)

# cmd = 'python generate_subarea_features.py'
# os.system(cmd)

cmd = 'python ../model/xgboost_navie_features.py'
os.system(cmd)
