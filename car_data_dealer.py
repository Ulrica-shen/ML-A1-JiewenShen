import random

import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, TensorDataset, random_split


def cleaned_dataset(df):

    # 6、列名name换成brand，并且只取第一个单词
    # 将列名name换成brand,inplace=True要写，不然不起作用
    df.rename(columns={'name': 'brand'}, inplace=True)
    # 更换整列的值，方法一：
    # df['brand'] = df['brand'].apply(get_first_name)
    # 方法二：
    df.loc[:, 'brand'] = [name.split()[0] for name in df['brand']]

    # 8、去掉试驾车一行，即把owner一列数据为Test Drive Car的行去掉,
    # 测试时可以注释这个功能，否则看不到试驾车的owner变成5
    # df.drop(df[df['owner'] == 'Test Drive Car'].index)
    df = df[df['owner'].str.contains('Test Drive Car') == False]

    # 1、将owner一列数据换成数字12345
    owner_feature_dic = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4,
        'Test Drive Car': 5
    }
    # df.replace({'owner': owner_feature_dic}, inplace=True)
    df['owner'].replace(owner_feature_dic, inplace=True)

    # 2、删除fuel列数据为CNG或者LPG的行
    df = df[df['fuel'].str.contains("CNG|LPG") == False]

    # 3、mileage列去掉kmpl这个单位，并且转换成float,要处理缺失值
    # df.loc[:, 'mileage'] = [mileage.split()[0] if isinstance(mileage,str) else 0 for mileage in df['mileage']]
    # df['mileage'] = df['mileage'].astype(float)
    change_to_numerical('mileage', df)

    # 4、engine列去掉CC这个单位，并且转换成float
    change_to_numerical('engine', df)

    # 5、max_power列去掉bhp这个单位，并且转换成float
    change_to_numerical('max_power', df)

    # 7、删除torque（扭矩）一列数据
    # pop可以列数据并且把数据返回
    column_data = df.pop("torque")

    # 在数据框架的开头插入行号
    df.insert(loc=0, column='', value=range(0, len(df)))
    # 对座位数插补
    df['seats'].fillna(df['seats'].mode()[0], inplace=True)

    return df


def change_to_numerical(columns_name, df):
    # 固定值插补
    # df[columns_name].fillna(-1, inplace=True)

    df.loc[:, columns_name] = [item.split()[0] if isinstance(item, str) else item for item in df[columns_name]]
    df[columns_name] = df[columns_name].astype(float)
    # 用均值插补
    # df[columns_name].fillna(df[columns_name].mean(), inplace=True)
    # 用众数插补
    df[columns_name].fillna(df[columns_name].mode()[0], inplace=True)
    # 4.用中位数插补
    # df[columns_name].fillna(df[columns_name].median()[0], inplace=True)
    # 5.用前一个数据插补
    # df[columns_name].fillna(method='pad', inplace=True)
    # 6.用后一个数据插补
    # df[columns_name].fillna(method='bfill', inplace=True)
    # 7.用插值法插补
    # df[columns_name].fillna(method='linear', inplace=True)


def get_first_name(name):
    return name.split()[0]
