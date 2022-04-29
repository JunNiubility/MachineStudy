import numpy as np
from navie_bayes import BernoulliNavieBayes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
mpl.rcParams['font.size'] = 12  # 字体大小
plt.rcParams['axes.unicode_minus'] = False  ## 设置正常显示符号


def experiment(X, y, test_size, N):
    """
    根据测试集比例test_size和训练次数N进行训练和预测，并计算出准确率
    :param X:待训练数据的特征
    :param y:待训练数据的标签
    :param test_size:测试集比例
    :param N:训练次数
    :return:准确率
    """
    acc = np.empty(N)
    for i in range(N):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        clf = BernoulliNavieBayes()
        clf.train(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc[i] = accuracy_score(y_test, y_pred)
    return np.mean(acc)


if __name__ == '__main__':
    # 数据准备
    dataurl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
    X = np.genfromtxt(dataurl, delimiter=',', usecols=range(48))
    y = np.genfromtxt(dataurl, delimiter=',', usecols=[-1], dtype=np.int_)
    print(f'X.shape={X.shape}\ny.shape={y.shape}')
    X = np.where(X > 0, 1, 0)
    # 训练后预测准确率
    test_size = np.linspace(0.1, 0.9, 9)
    sizes = [experiment(X, y, test_size=i, N=1000) for i in test_size]
    # 作图
    fig = plt.figure(dpi=300)
    plt.plot(test_size, sizes, 'b-.', lw=1)
    plt.title('测试集比列与准确率的关系', fontdict={'size': 10})
    plt.xlabel('测试集比例')
    plt.ylabel('准确率')
    plt.savefig('测试集比列与准确率的关系.png', dpi=1000)
    plt.show()
