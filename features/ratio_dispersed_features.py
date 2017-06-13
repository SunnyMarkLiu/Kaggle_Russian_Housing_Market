#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
将连续型特征进行离散化分段处理

@author: MarkLiu
@time  : 17-5-28 下午1:38
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
# remove warnings
import warnings

warnings.filterwarnings('ignore')

# my own module
import data_utils


def dispersed_neighbourhood_ratio_features(conbined_data):
    """ 对 generate_neighbourhood_features 生成的特征进行离散化分段处理，同时删除原有过拟合的的 ratio 特征"""
    features = ['life_sq_ratio', 'kitch_sq_ratio', 'kitch_sq_vs_life_ratio', '0_6_all_age_ratio', '7_14_all_age_ratio',
                '0_17_all_age_ratio', '16_29_all_age_ratio', '0_13_all_age_ratio']

    for f in features:
        mingap = (conbined_data.life_sq.max() - conbined_data.life_sq.min()) / 10.0
        conbined_data[f] = np.round(conbined_data[f].values / np.maximum(mingap,1.0))
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

    conbined_data = dispersed_neighbourhood_ratio_features(conbined_data)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['id'] = train_id
    train['price_doc'] = train_price_doc
    test['id'] = test_id
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== dispersed some continuous ratio features =============="
    main()
