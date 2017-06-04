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

warnings.filterwarnings('ignore')

# my own module
import data_utils


def generate_region_timewindow_price(train, test):
    """按照地区统计时间窗内的价格的统计特征"""
    df_sub_area = train[['sub_area', 'year']].select_dtypes(include=['object']).copy()
    train['sub_area_str'] = train['sub_area']
    del train['sub_area']
    for c in df_sub_area:
        df_sub_area[c] = pd.factorize(df_sub_area[c])[0]

    train = pd.concat([train, df_sub_area], axis=1)

    df_sub_area = test[['sub_area', 'year']].select_dtypes(include=['object']).copy()
    test['sub_area_str'] = test['sub_area']
    del test['sub_area']
    for c in df_sub_area:
        df_sub_area[c] = pd.factorize(df_sub_area[c])[0]

    test = pd.concat([test, df_sub_area], axis=1)

    for static in ['mean', 'median']:
        df = train.groupby(['sub_area', 'year']).agg(static)['price_doc'].reset_index()
        sub_area_mean_price = df.pivot('sub_area', 'year', 'price_doc').reset_index().fillna(np.nan)
        sub_area_mean_price.columns = ['sub_area', '2011_year', '2012_year', '2013_year', '2014_year', '2015_year']

        # 添加2011年的平均价格
        for i in range(train.shape[0]):
            train['sub_area_2011_year_'+static+'_price'] = \
                sub_area_mean_price[sub_area_mean_price['sub_area'] == train.loc[i, 'sub_area']]['2011_year']
            train['sub_area_2012_year_'+static+'_price'] = \
                sub_area_mean_price[sub_area_mean_price['sub_area'] == train.loc[i, 'sub_area']]['2012_year']
            train['sub_area_2013_year_'+static+'_price'] = \
                sub_area_mean_price[sub_area_mean_price['sub_area'] == train.loc[i, 'sub_area']]['2013_year']
            train['sub_area_2014_year_'+static+'_price'] = \
                sub_area_mean_price[sub_area_mean_price['sub_area'] == train.loc[i, 'sub_area']]['2014_year']
            train['sub_area_2015_year_'+static+'_price'] = \
                sub_area_mean_price[sub_area_mean_price['sub_area'] == train.loc[i, 'sub_area']]['2015_year']

        for i in range(test.shape[0]):
            test['sub_area_2011_year_'+static+'_price'] = \
                sub_area_mean_price[sub_area_mean_price['sub_area'] == test.loc[i, 'sub_area']]['2011_year']
            test['sub_area_2012_year_'+static+'_price'] = \
                sub_area_mean_price[sub_area_mean_price['sub_area'] == test.loc[i, 'sub_area']]['2012_year']
            test['sub_area_2013_year_'+static+'_price'] = \
                sub_area_mean_price[sub_area_mean_price['sub_area'] == test.loc[i, 'sub_area']]['2013_year']
            test['sub_area_2014_year_'+static+'_price'] = \
                sub_area_mean_price[sub_area_mean_price['sub_area'] == test.loc[i, 'sub_area']]['2014_year']
            test['sub_area_2015_year_'+static+'_price'] = \
                sub_area_mean_price[sub_area_mean_price['sub_area'] == test.loc[i, 'sub_area']]['2015_year']

    return train, test

def main():
    print 'loading train and test datas...'
    train, test, _ = data_utils.load_data()
    print 'train:', train.shape, ', test:', test.shape

    train, test = generate_region_timewindow_price(train, test)

    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== apply time window generate some statistic features =============="
    main()
