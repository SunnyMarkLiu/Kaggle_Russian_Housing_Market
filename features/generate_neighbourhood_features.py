#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
关于 neighbourhood 的 features
@author: MarkLiu
@time  : 17-5-26 下午3:36
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


def gen_area_features(conbined_data):
    """构造周边面积相关特征"""
    # 人均周边所占面积
    conbined_data['per_raion_person_area'] = conbined_data['area_m'] / conbined_data['raion_popul'].astype(float)
    # 人均绿地面积（模型过拟合）
    # conbined_data['per_raion_person_green_zone_part '] = conbined_data['green_zone_part'] / conbined_data[
    #     'raion_popul'].astype(float)
    # 人均工业区面积（模型过拟合）
    # conbined_data['per_raion_person_indust_part'] = conbined_data['indust_part'] / conbined_data['raion_popul'].astype(float)
    # # 工业区面积所占比例（模型过拟合）
    # conbined_data['indust_area_m_ratio'] = conbined_data['indust_part'] / conbined_data['area_m'].astype(float)

    return conbined_data


def gen_school_features(conbined_data):
    """构造教育相关特征"""
    # 学龄前人口数所占比例 （模型过拟合）
    # conbined_data['children_preschool_popul_ratio'] = \
    #     conbined_data['children_preschool'] / conbined_data['raion_popul'].astype(float)
    # # 学前教育机构的座位数所缺失数目 （模型过拟合）
    # conbined_data['children_preschool_preschool_quota_gap'] = \
    #     conbined_data['children_preschool'] - conbined_data['preschool_quota']
    # 学前教育机构座位缺失比例 （模型过拟合）
    # conbined_data['children_preschool_preschool_quota_gap_ratio'] = \
    #     (conbined_data['children_preschool'] - conbined_data['preschool_quota']) / conbined_data['children_preschool'].astype(float)

    # 学龄人口数所占比例 （模型过拟合）
    # conbined_data['children_school_popul_ratio'] = \
    #     conbined_data['children_school'] / conbined_data['raion_popul'].astype(float)
    # 地区中学座位数所缺失数目（模型过拟合）
    # conbined_data['children_school_school_quota_gap'] = \
    #     conbined_data['children_school'] - conbined_data['school_quota']
    # 地区中学座位数缺失比例（模型过拟合）
    # conbined_data['children_school_school_quota_gap_ratio'] = \
    #     (conbined_data['children_school'] - conbined_data['school_quota']) / conbined_data['children_school'].astype(float)

    # # 教育机构总数（模型过拟合）
    # conbined_data['total_education_counts'] = conbined_data['preschool_education_centers_raion'] + \
    #     conbined_data['school_education_centers_raion'] + conbined_data['sport_objects_raion'] + conbined_data['additional_education_raion']
    # # 高中院校数所占比例（模型过拟合）
    # conbined_data['school_education_centers_raion_ratio'] = \
    #     conbined_data['school_education_centers_raion'] / conbined_data['total_education_counts'].astype(float)

    # 学前教育机构所占比例（模型过拟合）
    # conbined_data['preschool_education_centers_raion_ratio'] = \
    #     conbined_data['preschool_education_centers_raion']  / conbined_data['total_education_counts'].astype(float)
    # 高等教育机构所占比例（模型过拟合）
    # conbined_data['sport_objects_raion_ratio'] = \
    #     conbined_data['sport_objects_raion']  / conbined_data['total_education_counts'].astype(float)
    # 额外教育机构所占比例（模型过拟合）
    # conbined_data['additional_education_raion_ratio'] = \
    #     conbined_data['additional_education_raion']  / conbined_data['total_education_counts'].astype(float)

    return conbined_data


def generate_hospital_features(conbined_data):
    """构造医院相关特征"""
    # 各区医院病床数目相对人口的比例（模型过拟合）
    # conbined_data['hospital_beds_raion_raion_popul_ratio'] = \
    #     conbined_data['hospital_beds_raion'] / conbined_data['raion_popul'].astype(float)
    return conbined_data


def generate_population_features(conbined_data):
    """构造全市人口相关特征（模型过拟合）"""

    # # 全市男性人口比例（模型过拟合）
    # conbined_data['male_ratio'] = conbined_data['male_f'] / conbined_data['full_all'].astype(float)
    # # 全市女性人口比例（模型过拟合）
    # conbined_data['female_ratio'] = conbined_data['female_f'] / conbined_data['full_all'].astype(float)

    # 小于工作年龄的人口比例（模型过拟合）
    # conbined_data['young_underwork_vs_full_all_ratio'] = conbined_data['young_all'] / conbined_data['full_all'].astype(float)
    # 小于工龄的男性人口比例（模型过拟合）
    conbined_data['young_male_vs_underwork_ratio'] = conbined_data['young_male'] / conbined_data['young_all'].astype(float)
    # 小于工龄的女性人口比例（模型过拟合）
    # conbined_data['young_female_vs_underwork_ratio'] = conbined_data['young_female'] / conbined_data['young_all'].astype(float)
    # # 男性中小于工龄的人口比例（模型过拟合）

    # conbined_data['young_male_vs_malef_ratio'] = conbined_data['young_male'] / conbined_data['male_f'].astype(float)

    # # 女性中小于工龄的人口比例（模型过拟合）
    # conbined_data['young_female_vs_femalef_ratio'] = conbined_data['young_female'] / conbined_data['female_f'].astype(float)

    # 劳动年龄人口比例（模型过拟合）
    # conbined_data['work_all_vs_full_all_ratio'] = conbined_data['work_all'] / conbined_data['full_all'].astype(float)
    # 劳动年龄男性人口比例（模型过拟合）
    conbined_data['work_male_vs_work_all_ratio'] = conbined_data['work_male'] / conbined_data['work_all'].astype(float)
    # # 劳动年龄女性人口比例（模型过拟合）
    # conbined_data['work_female_vs_work_all_ratio'] = conbined_data['work_female'] / conbined_data['work_all'].astype(float)
    # # 男性中劳动年龄的人口比例（模型过拟合）

    # conbined_data['work_male_vs_malef_ratio'] = conbined_data['work_male'] / conbined_data['male_f'].astype(float)

    # # 女性中劳动年龄的人口比例（模型过拟合）
    # conbined_data['work_female_vs_femalef_ratio'] = conbined_data['work_female'] / conbined_data['female_f'].astype(float)

    # 年龄大于工作年龄的人口比例（模型过拟合）
    # conbined_data['ekder_all_vs_full_all_ratio'] = conbined_data['ekder_all'] / conbined_data['full_all'].astype(float)
    # 年龄大于工作年龄的人口男性人口比例（模型过拟合）
    conbined_data['ekder_male_vs_ekder_all_ratio'] = conbined_data['ekder_male'] / conbined_data['ekder_all'].astype(float)
    # # 年龄大于工作年龄的人口女性人口比例（模型过拟合）
    # conbined_data['ekder_female_vs_ekder_all_ratio'] = conbined_data['ekder_female'] / conbined_data['ekder_all'].astype(float)
    # # 男性中年龄大于工作年龄的比例（模型过拟合）

    # conbined_data['ekder_male_vs_malef_ratio'] = conbined_data['ekder_male'] / conbined_data['male_f'].astype(float)

    # # 女性中年龄大于工作年龄的比例（模型过拟合）
    # conbined_data['ekder_female_vs_femalef_ratio'] = conbined_data['ekder_female'] / conbined_data['female_f'].astype(float)

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

    conbined_data = gen_area_features(conbined_data)
    conbined_data = gen_school_features(conbined_data)
    conbined_data = generate_hospital_features(conbined_data)
    conbined_data = generate_population_features(conbined_data)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['id'] = train_id
    train['price_doc'] = train_price_doc
    test['id'] = test_id
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== generate some neighbourhood features =============="
    main()
