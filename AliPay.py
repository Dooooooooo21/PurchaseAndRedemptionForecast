#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/10/26 21:40
# @Author  : dly
# @File    : AliPay.py
# @Desc    :

import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

from typing import *
import warnings

warnings.filterwarnings('ignore')

dataset_path = 'C:/Users/Dooooooooo21/Desktop/project/ALIPAY/Purchase Redemption Data/'

# 为方面后面操作，设置全局index变量
labels = ['total_purchase_amt', 'total_redeem_amt']
date_indexs = ['week', 'year', 'month', 'weekday', 'day']


# 加载数据
def load_data(path: str = 'user_balance_table.csv') -> pd.DataFrame:
    data_balance = pd.read_csv(path)
    return data_balance.reset_index(drop=True)


# 添加时间戳
def add_timestamp(data: pd.DataFrame, time_index: str = 'report_date') -> pd.DataFrame:
    data_balance = data.copy()
    data_balance['date'] = pd.to_datetime(data_balance[time_index], format="%Y%m%d")
    data_balance['day'] = data_balance['date'].dt.day
    data_balance['month'] = data_balance['date'].dt.month
    data_balance['year'] = data_balance['date'].dt.year
    data_balance['week'] = data_balance['date'].dt.week
    data_balance['weekday'] = data_balance['date'].dt.weekday
    return data_balance.reset_index(drop=True)


# 每日购买和赎回
def get_total_balance(data: pd.DataFrame, date: str = '2014-03-31') -> pd.DataFrame:
    df_tmp = data.copy()
    df_tmp = df_tmp.groupby(['date'])['total_purchase_amt', 'total_redeem_amt'].sum()
    df_tmp.reset_index(inplace=True)
    return df_tmp[(df_tmp['date'] >= date)].reset_index(drop=True)


# 测试数据
def generate_test_data(data: pd.DataFrame) -> pd.DataFrame:
    total_balance = data.copy()
    start = datetime.datetime(2014, 9, 1)
    testdata = []
    while start != datetime.datetime(2014, 10, 15):
        temp = [start, np.nan, np.nan]
        testdata.append(temp)
        start += datetime.timedelta(days=1)
    testdata = pd.DataFrame(testdata)
    testdata.columns = total_balance.columns

    total_balance = pd.concat([total_balance, testdata], axis=0)
    total_balance = total_balance.reset_index(drop=True)
    return total_balance.reset_index(drop=True)


# 加载用户信息
def load_user_information(path: str = 'user_profile_table.csv') -> pd.DataFrame:
    return pd.read_csv(path)


# 读取数据集
balance_data = load_data(dataset_path + 'user_balance_table.csv')
balance_data = add_timestamp(balance_data, time_index='report_date')
total_balance = get_total_balance(balance_data)
total_balance = generate_test_data(total_balance)
total_balance = add_timestamp(total_balance, 'date')
user_information = load_user_information(dataset_path + 'user_profile_table.csv')


# 特征提取

# 获取节假日集合

def get_holiday_set() -> Set[datetime.date]:
    holiday_set = set()
    # 清明节
    holiday_set = holiday_set | {datetime.date(2014, 4, 5), datetime.date(2014, 4, 6), datetime.date(2014, 4, 7)}
    # 劳动节
    holiday_set = holiday_set | {datetime.date(2014, 5, 1), datetime.date(2014, 5, 2), datetime.date(2014, 5, 3)}
    # 端午节
    holiday_set = holiday_set | {datetime.date(2014, 5, 31), datetime.date(2014, 6, 1), datetime.date(2014, 6, 2)}
    # 中秋节
    holiday_set = holiday_set | {datetime.date(2014, 9, 6), datetime.date(2014, 9, 7), datetime.date(2014, 9, 8)}
    # 国庆节
    holiday_set = holiday_set | {datetime.date(2014, 10, 1), datetime.date(2014, 10, 2), datetime.date(2014, 10, 3), \
                                 datetime.date(2014, 10, 4), datetime.date(2014, 10, 5), datetime.date(2014, 10, 6), \
                                 datetime.date(2014, 10, 7)}
    # 中秋节
    holiday_set = holiday_set | {datetime.date(2013, 9, 19), datetime.date(2013, 9, 20), datetime.date(2013, 9, 21)}
    # 国庆节
    holiday_set = holiday_set | {datetime.date(2013, 10, 1), datetime.date(2013, 10, 2), datetime.date(2013, 10, 3), \
                                 datetime.date(2013, 10, 4), datetime.date(2013, 10, 5), datetime.date(2013, 10, 6), \
                                 datetime.date(2013, 10, 7)}
    return holiday_set


# 提取所有 is特征
def extract_is_feature(data: pd.DataFrame) -> pd.DataFrame:
    total_balance = data.copy().reset_index(drop=True)

    # 是否是Weekend
    total_balance['is_weekend'] = 0
    total_balance.loc[total_balance['weekday'].isin((5, 6)), 'is_weekend'] = 1
    # 是否是假期
    total_balance['is_holiday'] = 0
    total_balance.loc[total_balance['date'].isin(get_holiday_set()), 'is_holiday'] = 1

    # 是否是节假日的第一天
    last_day_flag = 0
    total_balance['is_firstday_of_holiday'] = 0
    for index, row in total_balance.iterrows():
        if last_day_flag == 0 and row['is_holiday'] == 1:
            total_balance.loc[index, 'is_firstday_of_holiday'] = 1
        last_day_flag = row['is_holiday']

    # 是否是节假日的最后一天
    total_balance['is_lastday_of_holiday'] = 0
    for index, row in total_balance.iterrows():
        if row['is_holiday'] == 1 and total_balance.loc[index + 1, 'is_holiday'] == 0:
            total_balance.loc[index, 'is_lastday_of_holiday'] = 1

    # 是否是节假日后的上班第一天
    total_balance['is_firstday_of_work'] = 0
    last_day_flag = 0
    for index, row in total_balance.iterrows():
        if last_day_flag == 1 and row['is_holiday'] == 0:
            total_balance.loc[index, 'is_firstday_of_work'] = 1
        last_day_flag = row['is_lastday_of_holiday']

    # 是否不用上班
    total_balance['is_work'] = 1
    total_balance.loc[(total_balance['is_holiday'] == 1) | (total_balance['is_weekend'] == 1), 'is_work'] = 0
    special_work_day_set = {datetime.date(2014, 5, 4), datetime.date(2014, 9, 28)}
    total_balance.loc[total_balance['date'].isin(special_work_day_set), 'is_work'] = 1

    # 是否明天要上班
    total_balance['is_gonna_work_tomorrow'] = 0
    for index, row in total_balance.iterrows():
        if index == len(total_balance) - 1:
            break
        if row['is_work'] == 0 and total_balance.loc[index + 1, 'is_work'] == 1:
            total_balance.loc[index, 'is_gonna_work_tomorrow'] = 1

    # 昨天上班了吗
    total_balance['is_worked_yestday'] = 0
    for index, row in total_balance.iterrows():
        if index <= 1:
            continue
        if total_balance.loc[index - 1, 'is_work'] == 1:
            total_balance.loc[index, 'is_worked_yestday'] = 1

    # 是否是放假前一天
    total_balance['is_lastday_of_workday'] = 0
    for index, row in total_balance.iterrows():
        if index == len(total_balance) - 1:
            break
        if row['is_holiday'] == 0 and total_balance.loc[index + 1, 'is_holiday'] == 1:
            total_balance.loc[index, 'is_lastday_of_workday'] = 1

    # 是否周日要上班
    total_balance['is_work_on_sunday'] = 0
    for index, row in total_balance.iterrows():
        if index == len(total_balance) - 1:
            break
        if row['weekday'] == 6 and row['is_work'] == 1:
            total_balance.loc[index, 'is_work_on_sunday'] = 1

    # 是否是月初第一天
    total_balance['is_firstday_of_month'] = 0
    total_balance.loc[total_balance['day'] == 1, 'is_firstday_of_month'] = 1

    # 是否是月初第二天
    total_balance['is_secday_of_month'] = 0
    total_balance.loc[total_balance['day'] == 2, 'is_secday_of_month'] = 1

    # 是否是月初
    total_balance['is_premonth'] = 0
    total_balance.loc[total_balance['day'] <= 10, 'is_premonth'] = 1

    # 是否是月中
    total_balance['is_midmonth'] = 0
    total_balance.loc[(10 < total_balance['day']) & (total_balance['day'] <= 20), 'is_midmonth'] = 1

    # 是否是月末
    total_balance['is_tailmonth'] = 0
    total_balance.loc[20 < total_balance['day'], 'is_tailmonth'] = 1

    # 是否是每个月第一个周
    total_balance['is_first_week'] = 0
    total_balance.loc[total_balance['week'] % 4 == 1, 'is_first_week'] = 1

    # 是否是每个月第一个周
    total_balance['is_second_week'] = 0
    total_balance.loc[total_balance['week'] % 4 == 2, 'is_second_week'] = 1

    # 是否是每个月第一个周
    total_balance['is_third_week'] = 0
    total_balance.loc[total_balance['week'] % 4 == 3, 'is_third_week'] = 1

    # 是否是每个月第四个周
    total_balance['is_fourth_week'] = 0
    total_balance.loc[total_balance['week'] % 4 == 0, 'is_fourth_week'] = 1

    return total_balance.reset_index(drop=True)


# 提取is特征到数据集
total_balance = extract_is_feature(total_balance)


# 编码翌日特征
def encode_data(data: pd.DataFrame, feature_name: str = 'weekday', encoder=OneHotEncoder()) -> pd.DataFrame():
    total_balance = data.copy()
    week_feature = encoder.fit_transform(np.array(total_balance[feature_name]).reshape(-1, 1)).toarray()
    week_feature = pd.DataFrame(week_feature,
                                columns=[feature_name + '_onehot_' + str(x) for x in range(len(week_feature[0]))])
    # featureWeekday = pd.concat([total_balance, week_feature], axis = 1).drop(feature_name, axis=1)
    featureWeekday = pd.concat([total_balance, week_feature], axis=1)
    return featureWeekday


# 编码翌日特征到数据集
total_balance = encode_data(total_balance)

# 生成is特征集合
feature = total_balance[[x for x in total_balance.columns if x not in date_indexs]]


# 绘制箱型图
def draw_boxplot(data: pd.DataFrame) -> None:
    f, axes = plt.subplots(7, 4, figsize=(18, 24))
    global date_indexs, labels
    count = 0
    for i in [x for x in data.columns if x not in date_indexs + labels + ['date']]:
        sns.boxenplot(x=i, y='total_purchase_amt', data=data, ax=axes[count // 4][count % 4])
        count += 1
    plt.show()


# draw_boxplot(feature)
purchase_feature_seems_useless = [
    # 样本量太少，建模时无效；但若确定这是一个有用规则，可以对结果做修正
    'is_work_on_sunday',
    # 中位数差异不明显
    'is_first_week'
]


# 画相关性热力图
def draw_correlation_heatmap(data: pd.DataFrame, way: str = 'pearson') -> None:
    feature = data.copy()
    plt.figure(figsize=(20, 10))
    plt.title('The ' + way + ' coleration between total purchase and each feature')
    sns.heatmap(feature[[x for x in feature.columns if x not in ['total_redeem_amt', 'date']]].corr(way),
                linecolor='white',
                linewidths=0.1,
                cmap="RdBu")
    plt.show()


# draw_correlation_heatmap(feature, 'spearman')
# 剔除相关性较低的特征
temp = np.abs(feature[[x for x in feature.columns
                       if x not in ['total_redeem_amt', 'date']]].corr('spearman')['total_purchase_amt'])
feature_low_correlation = list(set(temp[temp < 0.1].index))
