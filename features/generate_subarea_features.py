#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
根据 sub_area 统计各地区的相关特征
@author: MarkLiu
@time  : 17-6-5 下午5:51
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
# remove warnings
import warnings

from tqdm import tqdm

warnings.filterwarnings('ignore')

# my own module
import data_utils


def generate_subarea_population_density(conbined_data):
    """统计各地区的人口密度"""
    return conbined_data


def generate_subarea_building_density(conbined_data):
    """统计各地区的建筑密度"""
    return conbined_data


def main():
    print 'loading train and test datas...'
    train, test, _ = data_utils.load_data()
    print 'train:', train.shape, ', test:', test.shape

    train_id = train['id']
    train_price_doc = train['price_doc']
    train.drop(['id', 'price_doc'], axis=1, inplace=True)
    test_id = test['id']
    test.drop(['id'], axis=1, inplace=True)

    # 合并训练集和测试集
    conbined_data = pd.concat([train[test.columns.values], test])
    conbined_data.columns = test.columns.values
    conbined_data = conbined_data.reset_index()

    conbined_data = generate_subarea_population_density(conbined_data)
    conbined_data = generate_subarea_building_density(conbined_data)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['id'] = train_id
    train['price_doc'] = train_price_doc
    test['id'] = test_id
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== generate some sub_area features =============="
    main()
