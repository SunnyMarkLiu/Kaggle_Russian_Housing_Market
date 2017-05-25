#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-5-23 上午11:56
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import cPickle
import numpy as np
import pandas as pd

# my own module
from conf.configure import Configure
import data_utils
# remove warnings
import warnings
warnings.filterwarnings('ignore')


def perform_missing_data(dataframe, columns, value):
    """ 填充缺失数据 """
    for column in columns:
        dataframe.loc[dataframe[column].isnull(), column] = value


def perform_area_features(train, test):
    """ 处理 area 相关字段 """
    perform_missing_data(train, ['life_sq', 'full_sq', 'kitch_sq'], 0)
    perform_missing_data(test, ['life_sq', 'full_sq', 'kitch_sq'], 0)

    # 去除 life_sq > full_sq 和 kitch_sq > full_sq 的异常数据
    train = train[train['kitch_sq'] <= train['full_sq']]
    train = train[train['life_sq'] <= train['full_sq']]

    gap = 50
    # 去除训练集中出现的数据而测试集中没有出现的数据避免过拟合
    train = train[train['full_sq'] <= test['full_sq'].max() + gap]
    train = train[train['life_sq'] <= test['life_sq'].max() + gap]

    # 居住面积比例
    train['life_sq_ratio'] = train['life_sq'] / (train['full_sq'] + 1)
    # 添加面积比例
    train['life_sq_ratio'] = train['life_sq'] / (train['full_sq'] + 1)
    train['kitch_sq_ratio'] = train['kitch_sq'] / (train['full_sq'] + 1)
    train['kitch_sq_vs_life_ratio'] = train['kitch_sq'] / (train['life_sq'] + 1)

    test['life_sq_ratio'] = test['life_sq'] / (test['full_sq'] + 1)
    test['kitch_sq_ratio'] = test['kitch_sq'] / (test['full_sq'] + 1)
    test['kitch_sq_vs_life_ratio'] = test['kitch_sq'] / (test['life_sq'] + 1)

    # perform log1p
    train['full_sq'] = np.log1p(train['full_sq'])
    test['full_sq'] = np.log1p(test['full_sq'])
    train['life_sq'] = np.log1p(train['life_sq'])
    test['life_sq'] = np.log1p(test['life_sq'])

    return train, test


def perform_floor_features(train, test):
    """ 处理 floor 相关字段 """
    perform_missing_data(train, ['life_sq', 'full_sq', 'kitch_sq'], 0)
    perform_missing_data(test, ['life_sq', 'full_sq', 'kitch_sq'], 0)
    return train, test


def perform_material_features(train, test):
    """ 处理 floor 相关字段 """
    perform_missing_data(train, ['material'], 0)
    perform_missing_data(test, ['material'], 0)
    return train, test


def perform_build_year_features(train, test):
    """ 处理 build_year 相关字段 """
    perform_missing_data(train, ['build_year'], 0)
    perform_missing_data(test, ['build_year'], 0)
    train['build_year'] = train['build_year'].map(lambda x: int(x))
    test['build_year'] = test['build_year'].map(lambda x: int(x))
    # 去除训练集中 build_year 异常的数据
    train = train[train['build_year'] <= test['build_year'][test['build_year'] > 0].max()]
    return train, test


def perform_num_room_features(train, test):
    """ 处理 num_room 相关字段 """
    perform_missing_data(train, ['num_room'], -1)
    perform_missing_data(test, ['num_room'], -1)
    # 对于 num_room 也视为缺失值
    train.loc[train['num_room'] == 0, 'num_room'] = -1
    test.loc[test['num_room'] == 0, 'num_room'] = -1

    # 添加每个 living rome 房间的面积
    train['per_living_room_sq'] = train['life_sq'] / train['num_room']
    test['per_living_room_sq'] = test['life_sq'] / test['num_room']

    return train, test


def perform_state_features(train, test):
    """ 处理 state 相关字段 """
    perform_missing_data(train, ['state'], -1)
    perform_missing_data(test, ['state'], -1)
    train['state'] = train['state'].map(lambda x: int(x))
    test['state'] = test['state'].map(lambda x: int(x))

    return train, test


def main():
    print 'loading train and test datas'
    print("Load data...")
    train, test = data_utils.load_data()

    print 'train:', train.shape, ', test:', test.shape

    print 'perform data cleaning and basic feature engineering'
    train['price_doc'] = np.log1p(train['price_doc'])

    train, test = perform_area_features(train, test)
    train, test = perform_floor_features(train, test)
    train, test = perform_material_features(train, test)
    train, test = perform_build_year_features(train, test)
    train, test = perform_num_room_features(train, test)
    print 'train:', train.shape, ', test:', test.shape


if __name__ == '__main__':
    print "================== train test preprocess =================="
    main()
