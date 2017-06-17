#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
1、发现不同地区价格存在不同的重复率
2、计算不同区间的价格的余弦相似度（特征的向量空间）

@author: MarkLiu
@time  : 17-6-17 下午3:35
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
# remove warnings
import warnings

warnings.filterwarnings('ignore')

# my own module
import data_utils


def perform_price_bins(train_price_doc, bins=50):
    """价格分段，用于计算余弦相似度的目标"""
    mingap = (max(train_price_doc) - min(train_price_doc)) / bins
    price_bins = [min(train_price_doc) + i * mingap for i in range(bins + 1)]
    return price_bins


def calc_cos_distance_pricebins(conbined_data_df, train_price_doc):
    """计算向量空间与各价格分段的余弦距离"""
    global complete_count
    conbined_data = conbined_data_df.copy()

    str_columns = conbined_data_df.select_dtypes(include=['object']).columns.values.tolist()
    str_columns.append('timestamp')
    conbined_data.drop(str_columns, axis=1, inplace=True)

    conbined_data.fillna(0, inplace=True)
    train_columns = conbined_data.columns.values
    print 'conbined_data:', conbined_data.shape

    scaler = preprocessing.StandardScaler().fit(conbined_data)
    conbined_data = scaler.transform(conbined_data)
    min_max_scaler = preprocessing.MinMaxScaler()
    conbined_data = min_max_scaler.fit_transform(conbined_data)
    conbined_data = pd.DataFrame(conbined_data, columns=train_columns)
    train = conbined_data.iloc[:train_price_doc.shape[0], :]

    price_bins = perform_price_bins(train_price_doc, bins=20)

    base_train_pbs = []
    for i in range(len(price_bins) - 1):
        left = price_bins[i]
        right = price_bins[i + 1]
        train_pb = train.loc[((train_price_doc > left) & (train_price_doc <= right)), :]
        base_train_pbs.append(train_pb)

    # 计算余弦相似度
    cos_distance_result = pd.DataFrame()
    cos_distance_result.index = range(cos_distance_result.shape[0])
    conbined_data.index = range(conbined_data.shape[0])

    for i in tqdm(range(conbined_data.shape[0])):
        for j, train_pb in enumerate(base_train_pbs):
            cos_distances = []
            train_pb = train_pb.values
            for k in range(train_pb.shape[0]):
                cos_dist = np.linalg.norm(conbined_data.loc[i, :] - train_pb[k])
                cos_distances.append(cos_dist)

            cos_distance_result.loc[i, 'prics_bin_' + str(j) + '_cos_distance'] = 1.0 * sum(cos_distances) / max(len(cos_distances), 1.0)

    return cos_distance_result


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
    conbined_data.index = range(conbined_data.shape[0])

    conbined_data_cos_dis = calc_cos_distance_pricebins(conbined_data, train_price_doc)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['id'] = train_id
    train['price_doc'] = train_price_doc
    test['id'] = test_id
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== perform cos_distance price bins =============="
    main()
