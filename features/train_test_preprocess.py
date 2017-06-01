#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-5-23 上午11:56
"""
import os
import sys

import pandas as pd

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np

# remove warnings
import warnings

warnings.filterwarnings('ignore')

# my own module
import data_utils
from impute_missing_data import simple_filling_missing_data


def perform_area_features(train, test):
    """ 处理 area 相关字段 """
    # 去除 life_sq > full_sq 和 kitch_sq > full_sq 的异常数据
    train = train[train['kitch_sq'] <= train['full_sq']]
    # train = train[train['life_sq'] <= train['full_sq']]

    gap = 50
    # 去除训练集中出现的数据而测试集中没有出现的数据避免过拟合
    train = train[train['full_sq'] <= test['full_sq'].max() + gap]
    train = train[train['life_sq'] <= test['life_sq'].max() + gap]

    # 添加面积比例
    # train['life_sq_ratio'] = train['life_sq'] / (train['full_sq'] + 1)
    # train['kitch_sq_ratio'] = train['kitch_sq'] / (train['full_sq'] + 1)
    # train['kitch_sq_vs_life_ratio'] = train['kitch_sq'] / (train['life_sq'] + 1)
    #
    # test['life_sq_ratio'] = test['life_sq'] / (test['full_sq'] + 1)
    # test['kitch_sq_ratio'] = test['kitch_sq'] / (test['full_sq'] + 1)
    # test['kitch_sq_vs_life_ratio'] = test['kitch_sq'] / (test['life_sq'] + 1)

    # perform log1p (有用)
    train['full_sq'] = np.log1p(train['full_sq'])
    test['full_sq'] = np.log1p(test['full_sq'])
    train['life_sq'] = np.log1p(train['life_sq'])
    test['life_sq'] = np.log1p(test['life_sq'])

    return train, test


def perform_floor_features(train, test):
    """ 处理 floor 相关字段 """
    train['floor'] = train['floor'].map(lambda f: int(round(f)))
    train['max_floor'] = train['max_floor'].map(lambda f: int(round(f)))
    test['floor'] = test['floor'].map(lambda f: int(round(f)))
    test['max_floor'] = test['max_floor'].map(lambda f: int(round(f)))

    train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)
    test['rel_floor'] = test['floor'] / test['max_floor'].astype(float)
    return train, test


def perform_material_features(train, test):
    """ 处理 floor 相关字段 """
    simple_filling_missing_data(train, ['material'], 0)
    simple_filling_missing_data(test, ['material'], 0)

    train['material'] = train['material'].map(lambda f: int(f))
    test['material'] = test['material'].map(lambda f: int(f))

    return train, test


def perform_build_year_features(train, test):
    """ 处理 build_year 相关字段 """
    train['build_year'] = train['build_year'].map(lambda x: int(x))
    test['build_year'] = test['build_year'].map(lambda x: int(x))
    # 去除训练集中 build_year 异常的数据
    train = train[train['build_year'] <= test['build_year'][test['build_year'] > 0].max()]
    return train, test


def perform_num_room_features(train, test):
    """ 处理 num_room 相关字段 """
    # 对于 num_room 也视为缺失值
    train.loc[train['num_room'] == 0, 'num_room'] = 2
    test.loc[test['num_room'] == 0, 'num_room'] = 2

    train['num_room'] = train['num_room'].map(lambda x: int(round(x)))
    test['num_room'] = test['num_room'].map(lambda x: int(round(x)))

    # 添加每个 living rome 房间的面积
    # train['per_living_room_sq'] = train['life_sq'] / train['num_room']
    # test['per_living_room_sq'] = test['life_sq'] / test['num_room']

    return train, test


def perform_state_features(train, test):
    """ 处理 state 相关字段 """
    simple_filling_missing_data(train, ['state'], -1)
    simple_filling_missing_data(test, ['state'], -1)
    train['state'] = train['state'].map(lambda x: int(x))
    test['state'] = test['state'].map(lambda x: int(x))
    return train, test


def perform_product_type_features(train, test):
    """ 处理 state 相关字段 """
    simple_filling_missing_data(train, ['product_type'], -1)
    simple_filling_missing_data(test, ['product_type'], -1)
    train['state'] = train['state'].map(lambda x: int(x))
    test['state'] = test['state'].map(lambda x: int(x))
    return train, test


def perform_round_int_features(train, test):
    """由于 kmeans 聚类求平均后有些属性变为float，需要将其round到int数值"""
    rounds_columns = ['raion_build_count_with_material_info', 'build_count_block',
                      'build_count_wood', 'build_count_frame', 'build_count_brick',
                      'build_count_monolith', 'build_count_panel', 'build_count_foam',
                      'build_count_slag', 'build_count_mix', 'build_count_before_1920',
                      'raion_build_count_with_builddate_info', 'build_count_1921-1945',
                      'build_count_1946-1970', 'build_count_1971-1995', 'build_count_after_1995',
                      'ID_railroad_station_walk', 'ID_railroad_station_avto', 'ID_big_road1',
                      'ID_big_road2']

    for column in rounds_columns:
        train[column] = train[column].map(lambda x: int(round(x)))
        test[column] = test[column].map(lambda x: int(round(x)))

    return train, test


def perform_timestamp_features(conbined_data):
    # # Add month-year
    # month_year = (conbined_data.timestamp.dt.month + conbined_data.timestamp.dt.year * 100)
    # month_year_cnt_map = month_year.value_counts().to_dict()
    # conbined_data['month_year_cnt'] = month_year.map(month_year_cnt_map)
    #
    # # Add week-year count
    # week_year = (conbined_data.timestamp.dt.weekofyear + conbined_data.timestamp.dt.year * 100)
    # week_year_cnt_map = week_year.value_counts().to_dict()
    # conbined_data['week_year_cnt'] = week_year.map(week_year_cnt_map)
    #
    # # Add month and day-of-week
    # conbined_data['month'] = conbined_data.timestamp.dt.month
    # conbined_data['dow'] = conbined_data.timestamp.dt.dayofweek

    # Remove timestamp column (may overfit the model in train)
    # conbined_data.drop(['timestamp'], axis=1, inplace=True)

    return conbined_data


def main():
    print 'loading train and test datas...'
    train, test, _ = data_utils.load_imputed_data()
    print 'train:', train.shape, ', test:', test.shape

    print 'perform data cleaning and basic feature engineering'
    train['price_doc'] = np.log1p(train['price_doc'])

    train, test = perform_area_features(train, test)
    train, test = perform_floor_features(train, test)
    train, test = perform_material_features(train, test)
    train, test = perform_build_year_features(train, test)
    train, test = perform_num_room_features(train, test)

    train_id = train['id']
    train_price_doc = train['price_doc']
    train.drop(['id', 'price_doc'], axis=1, inplace=True)
    test_id = test['id']
    test.drop(['id'], axis=1, inplace=True)

    # 合并训练集和测试集
    conbined_data = pd.concat([train[test.columns.values], test])
    conbined_data.columns = test.columns.values

    conbined_data = perform_timestamp_features(conbined_data)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['id'] = train_id
    train['price_doc'] = train_price_doc
    test['id'] = test_id
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "================== train test preprocess =================="
    main()
