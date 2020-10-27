#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 15:24
# @Author  : dly
# @File    : EDA.py
# @Desc    :

import pandas as pd
import numpy as np
import warnings
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from scipy import stats
from sklearn.preprocessing import OneHotEncoder

dataset_path = 'C:/Users/Dooooooooo21/Desktop/project/ALIPAY/Purchase Redemption Data/'
data_balance = pd.read_csv(dataset_path + 'user_balance_table.csv')

data_balance['date'] = pd.to_datetime(data_balance['report_date'], format='%Y%m%d')
data_balance['day'] = data_balance['date'].dt.day
data_balance['month'] = data_balance['date'].dt.month
data_balance['year'] = data_balance['date'].dt.year
data_balance['week'] = data_balance['date'].dt.week
data_balance['weekday'] = data_balance['date'].dt.weekday

# 聚合时间数据: 每天购买和赎回
total_balance = data_balance.groupby(['date'])['total_purchase_amt', 'total_redeem_amt'].sum().reset_index()

print(total_balance.head())

# 生成测试集区段数据
start = datetime.datetime(2014, 9, 1)
testdata = []
while start != datetime.datetime(2014, 10, 1):
    temp = [start, np.nan, np.nan]
    testdata.append(temp)
    start += datetime.timedelta(days=1)
testdata = pd.DataFrame(testdata)
testdata.columns = total_balance.columns

# 拼接数据集
total_balance = pd.concat([total_balance, testdata], axis=0)
print(total_balance.head())
print(total_balance.tail())

# 为数据集添加时间戳
total_balance['day'] = total_balance['date'].dt.day
total_balance['month'] = total_balance['date'].dt.month
total_balance['year'] = total_balance['date'].dt.year
total_balance['week'] = total_balance['date'].dt.week
total_balance['weekday'] = total_balance['date'].dt.weekday

# 画出每日总购买与赎回量的时间序列图
# fig = plt.figure(figsize=(20, 6))
# plt.plot(total_balance['date'], total_balance['total_purchase_amt'], label='purchase')
# plt.plot(total_balance['date'], total_balance['total_redeem_amt'], label='redeem')
#
# plt.legend(loc='best')
# plt.title("The lineplot of total amount of Purchase and Redeem from July.13 to Sep.14")
# plt.xlabel("Time")
# plt.ylabel("Amount")
# plt.show()

# 按翌日对数据聚合后取均值
total_balance_1 = total_balance[total_balance['date'] >= datetime.datetime(2014, 4, 1)]
week_sta = total_balance_1[['total_purchase_amt', 'total_redeem_amt', 'weekday']].groupby('weekday',
                                                                                          as_index=False).mean()

# 分析翌日的中位数特征
# plt.figure(figsize=(12, 5))
# ax = plt.subplot(1, 2, 1)
# plt.title('The barplot of average total purchase with each weekday')
# ax = sns.barplot(x="weekday", y="total_purchase_amt", data=week_sta, label='Purchase')
# ax.legend()
# ax = plt.subplot(1, 2, 2)
# plt.title('The barplot of average total redeem with each weekday')
# ax = sns.barplot(x="weekday", y="total_redeem_amt", data=week_sta, label='Redeem')
# ax.legend()
# plt.show()


# 使用OneHot方法将翌日特征划分，获取划分后特征
encoder = OneHotEncoder()
total_balance = total_balance.reset_index()
week_feature = encoder.fit_transform(np.array(total_balance['weekday']).reshape(-1, 1)).toarray()
week_feature = pd.DataFrame(week_feature, columns=['weekday_onehot'] * len(week_feature[0]))
feature = pd.concat([total_balance, week_feature], axis=1)[
    ['total_purchase_amt', 'total_redeem_amt', 'weekday_onehot', 'date']]
feature.columns = list(feature.columns[0:2]) + [x + str(i) for i, x in enumerate(feature.columns[2:-1])] + ['date']
