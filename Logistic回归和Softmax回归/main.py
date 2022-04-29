import numpy as np
from logistic import LogisticRegression
from softmax import SoftmaxRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
mpl.rcParams['font.size'] = 6  # 字体大小
plt.rcParams['axes.unicode_minus'] = False  ## 设置正常显示符号


def logistic_test(X, y):
    """
    logistic训练一次的预测准确率
    :param X:数据集输入
    :param y:数据集输出
    :return:准确率，迭代次数
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    ss = StandardScaler()
    ss.fit(X_train)
    X_train_std = ss.transform(X_train)
    X_test_std = ss.transform(X_test)
    step = clf_logistic.train(X_train_std, y_train)
    y_pred = clf_logistic.predict(X_test_std)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, step


def softmax_test(X, y):
    """
    softmax训练一次的预测准确率
    :param X:数据集输入
    :param y:数据集输出
    :return:准确率，迭代次数
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    ss = StandardScaler()
    ss.fit(X_train)
    X_train_std = ss.transform(X_train)
    X_test_std = ss.transform(X_test)
    clf_softmax.train(X_train_std, y_train)
    y_pred = clf_softmax.predict(X_test_std)
    accuracy, step = accuracy_score(y_test, y_pred)
    return accuracy, step


if __name__ == '__main__':
    # 获取数据
    X = np.genfromtxt('wine.data', delimiter=',', usecols=range(1, 14))
    y = np.genfromtxt('wine.data', delimiter=',', usecols=(0,))
    y -= 1
    eta = np.logspace(2, -5, 8)
    peta = []
    paccuracy = []
    prate = []
    for i in eta:
        clf_softmax = SoftmaxRegression(n_iter=10 ** 10, eta=i, tol=1e-4)
        clf_logistic = LogisticRegression(n_iter=10 ** 10, eta=i, tol=1e-4)
        # accuracy_mean = np.mean([softmax_test(X, y) for _ in range(50)])
        # print(f'softmax_accuracy={accuracy_mean}')
        X = X[y != 2]
        y = y[y != 2]
        accuracy_mean = np.mean([logistic_test(X, y) for _ in range(50)], axis=0)
        peta.append(i)
        paccuracy.append(accuracy_mean[0])
        prate.append(accuracy_mean[1])
        print(f'logistic_accuracy={accuracy_mean}')
    # 作图
    plt.figure(dpi=200)
    plt.plot(np.log10(peta), paccuracy, 'b-', lw=1)
    plt.title(f'步长对准确率的影响', fontdict={'size': 20})
    plt.xlabel('lg步长')
    plt.ylabel('准确率')
    plt.grid()
    plt.savefig('步长对准确率的影响.png')
    plt.show()
    plt.figure(dpi=200)
    plt.plot(np.log10(peta), prate, 'b-', lw=1)
    plt.title(f'步长对收敛速度的影响', fontdict={'size': 20})
    plt.xlabel('lg步长')
    plt.ylabel('收敛速度')
    plt.grid()
    plt.savefig('步长对收敛速度的影响.png')
    plt.show()
