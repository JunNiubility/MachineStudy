import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
mpl.rcParams['font.size'] = 12  # 字体大小
plt.rcParams['axes.unicode_minus'] = False  ## 设置正常显示符号


class GDLinearRegression:
    """
    梯度下降法
    """

    def __init__(self, n_iter=200, eta=1e-3, tol=None):
        """
        类构造函数
        :param n_iter:训练迭代次数
        :param eta:学习率
        :param tol:误差变化阈值
        """
        self.n_iter = n_iter
        self.eta = eta
        self.tol = tol
        # 模型参数w（训练时初始化）
        self.w = None

    @staticmethod
    def _loss(y, y_pred):
        """
        计算损失
        :param y:实际值
        :param y_pred:预测值
        :return:损失值
        """
        return np.sum((y_pred - y) ** 2) / y.size

    @staticmethod
    def _gradient(X, y, y_pred):
        """
        计算梯度
        :param X:训练数据输入
        :param y:训练数据输出
        :param y_pred:预测值
        :return:梯度
        """
        return np.matmul(y_pred - y, X) / y.size

    def _gradient_descent(self, w, X, y):
        """
        梯度下降算法
        :param w:权重
        :param X:训练数据输入
        :param y:训练数据输出
        :return:迭代次数
        """
        # 记录绘图数据
        px, py = [], []
        step_i = None
        plt.figure(dpi=200)
        # 若用户指定tol，则启用早期停止法
        if self.tol is not None:
            loss_old = np.inf
        # 使用梯度下降，至多迭代n_iter，更新w
        for step_i in range(self.n_iter):
            # 预测
            y_pred = self._predict(X, w)
            # 计算损失
            loss = self._loss(y, y_pred)
            px.append(step_i)
            py.append(loss)
            print('%4i Loss:%s' % (step_i, loss))
            # 动态画出loss走向
            if step_i < 50 or (step_i % 50 == 0 and step_i < 3000) or step_i % 1000 == 0:
                plt.clf()
                plt.plot(px, py, 'b-', lw=1)
                plt.title(f'损失函数的收敛曲线eta={self.eta}', fontdict={'size': 20})
                plt.text(step_i / 3, 3, 'Loss=%.4f' % loss, fontdict={'size': 20, 'color': 'red'})
                plt.pause(0.1)

            # 早期停止法
            if self.tol is not None:
                # 如果损失下降小于阈值，则终止迭代
                if loss_old - loss < self.tol:
                    break
                loss_old = loss
            # 计算梯度
            grad = self._gradient(X, y, y_pred)
            # 更新参数w
            w -= self.eta * grad
        fig = plt.gcf()
        fig.savefig(f'eta={self.eta}.png', dpi=200)
        plt.close(fig)
        return step_i

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
        :return:迭代次数
        """
        # 预处理X_train(添加x0=1)
        X_train = self._preprocess_data_X(X_train)
        # 初始化参数向量w
        _, n = X_train.shape
        self.w = np.random.random(n) * 0.05
        # 执行梯度下降训练w
        return self._gradient_descent(self.w, X_train, y_train)

    @staticmethod
    def _predict(X, w):
        """
        预测内部接口，实现函数h(x)
        :param X:待测数据输入
        :param w:训练权重
        :return:预测值
        """
        return np.matmul(X, w)

    def predict(self, X):
        """
        预测
        :param X:测试数据输入
        :return:预测数据输出
        """
        X = self._preprocess_data_X(X)
        return self._predict(X, self.w)
