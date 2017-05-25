#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-5-24 下午8:49
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
from collections import Counter

# my own module
import data_utils
from kmeans_impute_missing_data import KMeansImputeMissingData
# remove warnings
import warnings

warnings.filterwarnings('ignore')


def impute_categories_missing_data(dataframe, cate_columns):
    for column in cate_columns:
        most_common = Counter(dataframe[column].tolist()).most_common(1)[0][0]
        dataframe.loc[dataframe[column].isnull(), column] = most_common


def simple_filling_missing_data(dataframe, columns, value):
    """ 填充缺失数据 """
    for column in columns:
        if dataframe[column].isnull().sum() > 0:
            dataframe.loc[dataframe[column].isnull(), column] = value


def simple_impute_data(train, test):
    simple_filling_missing_data(train, ['build_year'], 0)
    simple_filling_missing_data(test, ['build_year'], 0)


def kmeans_impute_data(train, test):
    print '合并训练集和测试集'
    conbined_data = pd.concat([train[test.columns.values], test])
    conbined_data.columns = test.columns.values
    train_price_doc = train['price_doc']
    print train.shape, test.shape, conbined_data.shape

    num_columns = conbined_data.select_dtypes(exclude=['object']).columns.values.tolist()

    # 去除类别属性的数值类型的 column
    num_columns.remove('id')
    num_columns.remove('material')
    num_columns.remove('state')
    num_columns.remove('build_year')

    missing_rates = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    kmeans_impute_data = conbined_data[num_columns].copy()
    total_count = conbined_data.shape[0]
    for missing_rate in missing_rates:
        missing_df = conbined_data[num_columns].isnull().sum(axis=0).reset_index()
        missing_df.columns = ['column_name', 'missing_count']
        missing_df = missing_df[missing_df.missing_count > 0]

        if missing_df.shape[0] == 0:
            break

        missing_df = missing_df.sort_values(by='missing_count', ascending=False)
        missing_df['missing_rate'] = 1.0 * missing_df['missing_count'] / total_count
        if missing_df['missing_rate'].values[0] < missing_rate:
            continue

        print '填充缺失率大于{}的数据, 缺失数据属性 {}...'.format(missing_rate, missing_df.shape[0])
        # n_clusters 为超参数！
        impute_model = KMeansImputeMissingData(conbined_data[num_columns], n_clusters=20, max_iter=100)
        kmeans_labels, x_kmeans, centroids, global_labels, x_global_mean = impute_model.impute_missing_data()

        kmeans_impute_data[num_columns] = x_kmeans
        # 填充数据
        big_missing_columns = missing_df[missing_df.missing_rate > missing_rate]['column_name']
        conbined_data[big_missing_columns] = kmeans_impute_data[big_missing_columns]

        missing_df = conbined_data[num_columns].isnull().sum(axis=0).reset_index()
        missing_df.columns = ['column_name', 'missing_count']
        missing_df = missing_df[missing_df.missing_count > 0]
        print '缺失数据属性 {}'.format(missing_df.shape[0])

    print "缺失数值型数据填充完成"
    train = conbined_data.iloc[:train.shape[0], :]
    train['price_doc'] = train_price_doc
    test = conbined_data.iloc[train.shape[0]:, :]
    print train.shape, test.shape


def impute_categories_data(train, test):
    str_columns = train.select_dtypes(include=['object']).columns.values.tolist()
    impute_categories_missing_data(train, str_columns)
    impute_categories_missing_data(test, str_columns)


def main():
    print("Load data...")
    train, test = data_utils.load_data()
    print("填充缺失数据...")
    simple_impute_data(train, test)
    kmeans_impute_data(train, test)
    impute_categories_data(train, test)
    print("Save data...")
    data_utils.save_data(train, test)


if __name__ == '__main__':
    print "================== impute missing data =================="
    main()
