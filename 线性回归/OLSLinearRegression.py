import numpy as np


class OLSLinearRegression:
    """
    线性回归
    """

    def __init__(self):
        self.w = None

    @staticmethod
    def _ols(X, y):
        """
        最小二乘法估算w
        :param X:训练数据输入
        :param y:训练数据输出
        :return:估算w
        """
        tmp = np.linalg.inv(np.matmul(X.T, X))
        tmp = np.matmul(tmp, X.T)
        return np.matmul(tmp, y)

    @staticmethod
    def _preprocess_data_X(X):
        """
        数据预处理
        :param X:训练数据输入
        :return:训练数据处理后的数据
        """
        # 扩展X，添加x0列并设置为1
        m, n = X.shape
        X_ = np.empty((m, n + 1))
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X_

    def train(self, X_train, y_train):
        """
        训练模型
        :param X_train:测试数据输入
        :param y_train:测试数据输出
        :return:None
        """
        # 预处理X_train(添加x0=1)
        X_train = self._preprocess_data_X(X_train)
        # 使用最小二乘法估算w
        self.w = self._ols(X_train, y_train)

    def predict(self, X):
        """
        预测
        :param X:测试数据输入
        :return:预测数据输出
        """
        # 预处理X_train(添加x0=1)
        X = self._preprocess_data_X(X)
        return np.matmul(X, self.w)
