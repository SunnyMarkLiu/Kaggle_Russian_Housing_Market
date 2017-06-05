#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
将生成的 ratio 等特征进行离散化分段处理

@author: MarkLiu
@time  : 17-5-28 下午1:38
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
# remove warnings
import warnings

warnings.filterwarnings('ignore')

# my own module
import data_utils


def dispersed_neighbourhood_ratio_features(conbined_data):
    """ 对 generate_neighbourhood_features 生成的特征进行离散化分段处理，同时删除原有过拟合的的 ratio 特征"""

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
