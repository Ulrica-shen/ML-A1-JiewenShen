import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import pre_model
import numpy as np


# 清洗数据同时划分为训练集和测试集
def predict_price(train_df, predict_df):

    # 清洗数据
    _df = organize_dataset(train_df)
    _test_df = organize_dataset(predict_df)

    # 数据切分
    df_train = _df[~_df['selling_price'].isnull()]
    df_train = df_train.reset_index(drop=True)
    df_test = _df[_df['selling_price'].isnull()]
    # print(df_train.shape,df_test.shape)

    no_features = ['selling_price']
    # 输入特征列
    features = [col for col in df_train.columns if col not in no_features]

    X = df_train[features]  # 训练集输入
    y = df_train['selling_price']  # 训练集标签

    # X_test = df_test[features]  # 测试集输入
    X_test = _test_df[features]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2023)
    # print(X_train.shape, X_val.shape,y_train.shape, y_val.shape)

    # 线性回归
    pre_model.get_tv_score(X_train, y_train, X_val, y_val)
    print('--------------------------')
    # LGBM
    pre_df = pre_model.lgbm(X_train, y_train, X_val, y_val, X_test)

    # 替换原来的价格
    predict_df['selling_price'] = pre_df['price']
    return predict_df


# 数据处理
def organize_dataset(df):
    _df = df.copy(deep=True)
    # 粗处理，删除重复值，补充缺失值，删除不需要的name，和torque
    # 带单位的数字全部去掉单位并且转化成数字类型,删除fuel列数据为CNG或者LPG的行
    # 去掉试驾车一行，即把owner一列数据为Test Drive Car的行去掉
    _df = cleaned_data(_df)
    # 通过散点图去除一些异常数据
    _df = eda(_df)
    # 特征编码，对分类型的数据编码
    _df = feature_encode(_df)
    return _df


# 对分类型数据进行编码
def feature_encode(df):
    for feat in ['year', 'fuel', 'seller_type', 'transmission', 'owner']:
        lbl = LabelEncoder()
        lbl.fit(df[feat])
        df[feat] = lbl.transform(df[feat])
    return df


# 散点图
def eda(df):

    # 年份与价格
    # sns.scatterplot(x='year', y='selling_price',  data=df)
    # plt.figure(figsize=(9, 6))
    # 剔除异常点1983，1990
    df = df[df['year'] > 1991]
    # sns.scatterplot(x='year', y='selling_price', data=df)
    # plt.show()

    # 前任里程与价格,可以看到有两个点里程数超过150万
    # sns.scatterplot(y='km_driven', x='selling_price', data=df)
    # plt.xticks(rotation=60)
    # 剔除这两个异常点
    df = df[df['km_driven'] > 0]
    df = df[df['km_driven'] < 1000000]
    # plt.figure(figsize=(9, 6))
    # sns.scatterplot(x='km_driven', y='selling_price', data=df)
    # plt.show()

    # 燃料类型
    # sns.catplot(x='fuel', y='selling_price', jitter=True, height=6, aspect=2, data=df);
    # plt.show()

    return df


def cleaned_data(df):

    # 查看数据缺失情况，
    # df.info()
    # 获取缺失率--mileage , engine, seats,  max_power
    # all_data_na = (df.isnull().sum() / len(df)) * 100
    # all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    # missing_data = pd.DataFrame({'缺失率': all_data_na})
    # print(missing_data)

    # 缩小价格
    df.loc[:, 'selling_price'] = [c for c in np.log(df['selling_price'])]

    # 查看重复数据
    # print(df.duplicated().sum())
    # 去除重复值
    df.drop_duplicates(inplace=True)

    # 删除不需要的扭矩列
    # 删去缺失价格列和Name列
    df.drop("torque", axis=1, inplace=True)
    df.drop("name", axis=1, inplace=True)

    # 检查处理缺失变量----mileage , engine, seats,  max_power

    # 用中位数填充缺失值
    df['seats'] = df['seats'].fillna(df['seats'].median())
    # print(df['seats'].isnull().sum())

    # max_power
    # 去除单位
    # df['max_power'] = df['max_power'].str.split(" ", expand=True)[0]
    # # 转化为数字
    # df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
    # # 用中位数填充缺失值
    # df['max_power'] = df['max_power'].fillna(df['max_power'].median())
    # # print(df['max_power'].isnull().sum())

    deal_cols = ['mileage', 'engine', 'max_power']
    for col in deal_cols:
        df[col] = df[col].str.split(" ", expand=True)[0]
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
        # print(df[col].isnull().sum())

    # 客户特殊要求的处理
    # 删除fuel列数据为CNG或者LPG的行
    df = df[df['fuel'].str.contains("CNG|LPG") == False]
    # 去掉试驾车一行，即把owner一列数据为Test Drive Car的行去掉
    df = df[df['owner'].str.contains('Test Drive Car') == False]

    return df


# if __name__ == '__main__':
#     df = pd.read_csv('D:\\学习资料\\项目\\Cars.csv')
#     predict_df = pd.read_csv('D:\\学习资料\\项目\\testCars.csv')
#     pd.set_option('display.max_columns', None)
#     pd.set_option('expand_frame_repr', False)
#     cleaned_divide(df.copy(deep=True),predict_df.copy(deep=True))


