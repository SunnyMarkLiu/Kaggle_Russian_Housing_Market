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

    simple_filling_missing_data(train, ['state'], -1)
    simple_filling_missing_data(test, ['state'], -1)

    simple_filling_missing_data(train, ['material'], 0)
    simple_filling_missing_data(test, ['material'], 0)

    train['build_year'][train['build_year'] == 20052009] = 2005
    train['build_year'][train['build_year'] == 20] = 2000
    train['build_year'][train['build_year'] == 215] = 2015
    train['build_year'][train['build_year'] == 4965] = 1965
    train['build_year'][train['build_year'] == 71] = 1971

    test['build_year'][test['build_year'] == 20052009] = 2005
    test['build_year'][test['build_year'] == 20] = 2000
    test['build_year'][test['build_year'] == 215] = 2015
    test['build_year'][test['build_year'] == 4965] = 1965
    test['build_year'][test['build_year'] == 71] = 1971

    train['build_year'][train['build_year'] < 20] = None
    test['build_year'][test['build_year'] < 20] = None

    return train, test


def kmeans_impute_data(data_df, num_columns, missing_rates):
    kmeans_impute_data = data_df[num_columns].copy()
    total_count = data_df.shape[0]
    for missing_rate in missing_rates:
        missing_df = data_df[num_columns].isnull().sum(axis=0).reset_index()
        missing_df.columns = ['column_name', 'missing_count']
        missing_df = missing_df[missing_df.missing_count > 0]

        if missing_df.shape[0] == 0:  # 不存在缺失数据
            break

        missing_df = missing_df.sort_values(by='missing_count', ascending=False)
        missing_df['missing_rate'] = 1.0 * missing_df['missing_count'] / total_count
        if missing_df['missing_rate'].values[0] < missing_rate:
            continue

        # print '填充缺失率大于{}的数据, 缺失数据属性 {}...'.format(missing_rate, missing_df.shape[0])
        # n_clusters 为超参数！
        impute_model = KMeansImputeMissingData(data_df[num_columns], n_clusters=20, max_iter=100)
        kmeans_labels, x_kmeans, centroids, global_labels, x_global_mean = impute_model.impute_missing_data()

        kmeans_impute_data[num_columns] = x_kmeans
        # 填充数据
        big_missing_columns = missing_df[missing_df.missing_rate > missing_rate]['column_name']
        data_df[big_missing_columns] = kmeans_impute_data[big_missing_columns]

        missing_df = data_df[num_columns].isnull().sum(axis=0).reset_index()
        missing_df.columns = ['column_name', 'missing_count']
        missing_df = missing_df[missing_df.missing_count > 0]
        # print '缺失数据属性 {}'.format(missing_df.shape[0])

    return data_df


def kmeans_impute_train_test_data(train, test):
    conbined_data = pd.concat([train[test.columns.values], test])
    conbined_data.columns = test.columns.values
    train_price_doc = train['price_doc']

    num_columns = conbined_data.select_dtypes(exclude=['object']).columns.values.tolist()

    # 去除类别属性的数值类型的 column
    num_columns.remove('id')
    num_columns.remove('material')
    num_columns.remove('state')

    missing_rates = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    conbined_data = kmeans_impute_data(conbined_data, num_columns, missing_rates)

    train = conbined_data.iloc[:train.shape[0], :]
    train['price_doc'] = train_price_doc
    test = conbined_data.iloc[train.shape[0]:, :]
    return train, test


def impute_categories_data(train, test):
    str_columns = train.select_dtypes(include=['object']).columns.values.tolist()
    impute_categories_missing_data(train, str_columns)
    impute_categories_missing_data(test, str_columns)
    return train, test


def contains_null(dataframe):
    missing_df = dataframe.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df[missing_df.missing_count > 0]
    return missing_df.shape[0] > 0


def impute_macro_data(macro):
    # child_on_acc_pre_school:等待学前教育机构确定的儿童数量
    simple_filling_missing_data(macro, ['child_on_acc_pre_school'], 'None')
    macro['child_on_acc_pre_school'][macro['child_on_acc_pre_school'] == '#!'] = -1
    macro['child_on_acc_pre_school'][macro['child_on_acc_pre_school'] == 'None'] = -1
    macro['child_on_acc_pre_school'][macro['child_on_acc_pre_school'] == '45,713'] = 45713
    macro['child_on_acc_pre_school'][macro['child_on_acc_pre_school'] == '7,311'] = 7311
    macro['child_on_acc_pre_school'][macro['child_on_acc_pre_school'] == '3,013'] = 3013
    macro['child_on_acc_pre_school'][macro['child_on_acc_pre_school'] == '16,765'] = 16765

    simple_filling_missing_data(macro, ['modern_education_share'], 'None')
    macro['modern_education_share'][macro['modern_education_share'] == 'None'] = -1
    macro['modern_education_share'][macro['modern_education_share'] == '90,92'] = 9092
    macro['modern_education_share'][macro['modern_education_share'] == '93,08'] = 9308
    macro['modern_education_share'][macro['modern_education_share'] == '95,4918'] = 954918

    simple_filling_missing_data(macro, ['old_education_build_share'], 'None')
    macro['old_education_build_share'][macro['old_education_build_share'] == 'None'] = -1
    macro['old_education_build_share'][macro['old_education_build_share'] == '23,14'] = 2314
    macro['old_education_build_share'][macro['old_education_build_share'] == '25,47'] = 2547
    macro['old_education_build_share'][macro['old_education_build_share'] == '8,2517'] = 82517

    num_columns = macro.select_dtypes(exclude=['object']).columns.values.tolist()
    missing_rates = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    macro = kmeans_impute_data(macro, num_columns, missing_rates)

    return macro


def main():
    print("Load data...")
    train, test, macro = data_utils.load_for_impute_data()
    print train.shape, test.shape, macro.shape

    if contains_null(train) | contains_null(test) | contains_null(macro):
        print("填充 train, test 缺失数据...")
        train, test = simple_impute_data(train, test)
        train, test = kmeans_impute_train_test_data(train, test)
        train, test = impute_categories_data(train, test)
        print("填充 macro 缺失数据...")
        macro = impute_macro_data(macro)
        print("缺失数据填充完成")
        print("Save data...")
        print train.shape, test.shape, macro.shape
        data_utils.save_imputed_data(train, test, macro)
    else:
        print "没有缺失数据!"


if __name__ == '__main__':
    print "================== impute missing data =================="
    main()
