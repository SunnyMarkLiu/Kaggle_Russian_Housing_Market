#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
根据时间窗按照地区统计年、月、季度的销量、平均价格等统计特征
@author: MarkLiu
@time  : 17-6-4 上午11:07
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
# remove warnings
import warnings

from tqdm import tqdm

warnings.filterwarnings('ignore')

import datetime

# my own module
import data_utils


def generate_timewindow_salecount(conbined_data):
    """按照地区统计时间窗内的销售量"""
    timewindow_days = [30*6, 30*4, 30*2, 30, 20, 10]

    for timewindow in timewindow_days:
        print 'perform timewindow =', timewindow
        pre_timewindow_salecounts = []
        for i in tqdm(range(conbined_data.shape[0])):
            today_time = conbined_data.loc[i, 'timestamp']
            indexs = (today_time - datetime.timedelta(days=timewindow) < conbined_data['timestamp']) & \
                     (conbined_data['timestamp'] < today_time)
            df = conbined_data[indexs]
            df = df.groupby(['sub_area']).count()['timestamp'].reset_index()
            df.columns = ['sub_area', 'sale_count']

            sale_count = df[df['sub_area'] == conbined_data.loc[i, 'sub_area']]['sale_count'].values
            sale_count = 0 if len(sale_count) == 0 else sale_count[0]
            pre_timewindow_salecounts.append(sale_count)

        conbined_data['this_sub_area_pre_'+str(timewindow)+'_salecount'] = pre_timewindow_salecounts

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

    conbined_data = generate_timewindow_salecount(conbined_data)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['id'] = train_id
    train['price_doc'] = train_price_doc
    test['id'] = test_id
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== apply time window generate some statistic features =============="
    main()
