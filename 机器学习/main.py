import numpy as np
import pandas as pd
import re
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


def data_process(url, **args):
    """
    数据预处理
    :param url: 文件地址
    :param args:
    :return: 预处理后的数据
    """
    data = pd.read_csv(url, **args)
    # 去除缺省值
    droplist = []
    for i in range(len(data)):
        if '-' not in np.array(data.iloc[i]):
            droplist.append(i)
    data = data.iloc[droplist]

    # 处理异常值
    passenger = []
    for items in data['rated_passenger']:
        passenger.append(re.findall(r"\d+", items)[0])
    data['rated_passenger'] = passenger

    # 产生新特征
    mean_price, price_diff, price_new = [], [], []
    for price in data['price_level']:
        price_data = re.findall(r"\d+", price)
        price_new = [int(x) for x in price_data]
        mean_price.append(np.mean(price_new))
        if len(price_new) == 2:
            price_diff.append(price_new[1] - price_new[0])
        else:
            price_diff.append(0)
    data['mean-price'] = mean_price
    data['price-diff'] = price_diff

    # 分类特征编码
    le = LabelEncoder()
    data['if_charging'] = le.fit_transform(data['if_charging'])
    data['gearbox_type'] = le.fit_transform(data['gearbox_type'])
    data['price_level']=le.fit_transform(data['price_level'])
    return data


if __name__ == '__main__':
    # 导入数据并进行预处理
    train_sales = data_process(url='./data/train_sales.csv', encoding='ISO-8859-1',low_memory=False)
    train_sales.info()
    X = train_sales.drop(labels=['sale_quantity','price_level'],axis=1)
    y = train_sales['sale_quantity']

    # 划分成训练样本和测试样本
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 使用随机森林(random forest)回归(RandomForestRegressor)模型
    forest = RandomForestRegressor(
        n_estimators=100,
        random_state=1,
        n_jobs=-1)
    # 模型训练
    forest.fit(X_train, y_train)
    #模型预测
    y_pred = forest.predict(X_test)
    # 模型评价
    mae = mean_absolute_error(y_test.values, y_pred)

    # 预测
    # 导入数据并进行预处理
    test_sales=data_process(url='./data/test_sales.csv', encoding='ISO-8859-1',low_memory=False)
    test = test_sales.drop(labels=['price_level'], axis=1)
    # 预测并进行四舍五入
    test_pred=np.round(forest.predict(test),decimals=0)
    # 准备导出数据
    submit={'sale_id':test_sales['sale_id'],'sale_quantity':test_pred}
    submit=pd.DataFrame(submit)
    submit.to_csv(r'./submit/submit_sales.csv',index=False)

    # 打印信息
    # 打印mean-priced相关信息
    print('the information of the mean-priced:\n',train_sales['mean-price'].head())
    print('the shape of the mean-priced:',np.shape(train_sales['mean-price'].tolist()))
    # 打印price-diff相关信息
    print('the information of the price-diff:\n', train_sales['price-diff'].head())
    print('the shape of the price-diff:', np.shape(train_sales['price-diff'].tolist()))
    # 打印新特征表信息
    print('新特征表信息：\n',train_sales.head())
    print(train_sales.shape)
    # 打印预测表信息
    print('打印预测表信息：\n', submit.head())
    print(submit.shape)