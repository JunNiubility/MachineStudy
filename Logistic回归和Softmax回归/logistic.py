import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
mpl.rcParams['font.size'] = 10  # 字体大小
plt.rcParams['axes.unicode_minus'] = False  ## 设置正常显示符号


class LogisticRegression:
    """
    逻辑回归
    """

    def __init__(self, n_iter=200, eta=1e-3, tol=None):
        """
        构造函数
        :param n_iter:迭代运行次数
        :param eta:学习率
        :param tol:误差变化阈值
        """
        self.n_iter = n_iter
        self.eta = eta
        self.tol = tol
        # 模型参数w(训练时初始化)
        self.w = None

    @staticmethod
    def _z(X, w):
        """
        g(x)函数:计算 x与w的内积
        :param X:训练数据输入
        :param w:权重
        :return:x与w的内积
        """
        return np.dot(X, w)

    @staticmethod
    def _sigmoid(z):
        """
        Logistic函数
        :param z:直线方程
        :return:Logistic函数值
        """
        if np.all(z >= 0):  # 对sigmoid函数优化，避免出现极大的数据溢出
            return 1. / (1. + np.exp(-z))
        else:
            return np.exp(z) / (1. + np.exp(z))

    def _predict_proba(self, X, w):
        """
        h(x)函数:预测为正例(y=1)的概率
        :param X:训练数据输入
        :param w:权重
        :return:正例(y=1)的概率
        """
        z = self._z(X, w)
        return self._sigmoid(z)

    @staticmethod
    def _loss(y, y_proba):
        """
        计算损失
        :param y:训练数据输出
        :param y_proba:正例(y=1)的概率
        :return:损失值
        """
        m = y.size
        p = y_proba * (2 * y - 1) + (1 - y)
        return -np.sum(np.log(p)) / m

    @staticmethod
    def _gradient(X, y, y_proba):
        """
        计算梯度
        :param X:训练数据输入
        :param y:训练数据输出
        :param y_proba:正例(y=1)的概率
        :return:梯度
        """
        return np.matmul(y_proba - y, X) / y.size

    def _gradient_descent(self, w, X, y):
        """
        梯度下降算法
        :param w:权重
        :param X:训练数据输入
        :param y:训练数据输出
        :return:迭代次数
        """
        print(f'当前步长为{self.eta}')
        # 若用户指定tol，则启用早期停止法
        if self.tol is not None:
            loss_old = np.inf
        # 使用梯度下降，至多迭代n_iter次，更新w
        step_i = None
        px, py = [], []
        plt.figure(dpi=200)
        for step_i in range(self.n_iter):
            # 预测所有点为1的概率
            y_proba = self._predict_proba(X, w)
            # 计算损失
            loss = self._loss(y, y_proba)
            print('%4i Loss: %s' % (step_i, loss))
            px.append(step_i)
            py.append(loss)
            # 动态画出loss走向
            if step_i < 50 or (step_i % 50 == 0 and step_i < 1000) or step_i % 1000 == 0:
                plt.clf()
                plt.plot(px, py, 'b-', lw=1)
                plt.title(f'损失函数的收敛曲线eta={self.eta}', fontdict={'size': 20})
                plt.text(step_i / 3, loss, 'Loss=%.4f' % loss, fontdict={'size': 20, 'color': 'red'})
                plt.pause(0.1)

            # 早期停止法
            if self.tol is not None:
                # 如果损失下降小于阈值，则终止迭代
                if loss_old - loss < self.tol:
                    break
                loss_old = loss

            # 计算梯度
            grad = self._gradient(X, y, y_proba)
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
        :return:处理后的训练数据输入
        """
        # 扩展X,添加x0列并设置为1
        m, n = X.shape
        X_ = np.empty([m, n + 1])
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X

    def train(self, X_train, y_train):
        """
        训练
        :param X_train:训练数据输入
        :param y_train:训练数据输出
        :return:迭代次数
        """
        # 预处理X_train(添加x0=1)
        X_train = self._preprocess_data_X(X_train)
        # 初始化参数向量w
        _, n = X_train.shape
        self.w = np.random.random(n) * 0.05
        # 执梯度下降训练w
        return self._gradient_descent(self.w, X_train, y_train)

    def predict(self, X):
        """
        预测
        :param X:测试数据输入
        :return:预测值输出
        """
        # 预处理X_test(x0=1)
        X = self._preprocess_data_X(X)
        # 预测为正例(y=1)的概率
        y_pred = self._predict_proba(X, self.w)
        # 根据概率预测类别，p>=0.5为正例，否则为负例
        return np.where(y_pred >= 0.5, 1, 0)
