#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-1 下午3:05
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


def delete_some_almost_zero_variance_features(conbined_data, macro):
    """ Delete Variables with almost zero variance"""
    almost_zero_variance = ["culture_objects_top_25_raion", "oil_chemistry_raion",
                            "railroad_terminal_raion", "nuclear_reactor_raion",
                            "build_count_foam", "big_road1_1line",
                            "railroad_1line", "office_sqm_500",
                            "trc_sqm_500", "cafe_count_500_price_4000",
                            "cafe_count_500_price_high", "mosque_count_500",
                            "leisure_count_500", "office_sqm_1000",
                            "trc_sqm_1000", "cafe_count_1000_price_high",
                            "mosque_count_1000", "cafe_count_1500_price_high",
                            "mosque_count_1500", "cafe_count_2000_price_high"]

    for c in almost_zero_variance:
        if c in conbined_data.columns.values:
            del conbined_data[c]
        if c in macro.columns.values:
            del macro[c]

    return conbined_data, macro


def main():
    print 'loading train and test datas...'
    train, test, macro = data_utils.load_data()
    print 'train:', train.shape, ', test:', test.shape

    train_id = train['id']
    train_price_doc = train['price_doc']
    train.drop(['id', 'price_doc'], axis=1, inplace=True)
    test_id = test['id']
    test.drop(['id'], axis=1, inplace=True)

    # 合并训练集和测试集
    conbined_data = pd.concat([train[test.columns.values], test])
    conbined_data.columns = test.columns.values

    conbined_data, macro = delete_some_almost_zero_variance_features(conbined_data, macro)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['id'] = train_id
    train['price_doc'] = train_price_doc
    test['id'] = test_id
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, macro)


if __name__ == '__main__':
    print "============== generate some statistic features =============="
    main()
