import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from GDLinearRegression import GDLinearRegression
import matplotlib.pyplot as plt
import matplotlib as mpl

# from OLSLinearRegression import OLSLinearRegression
mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
mpl.rcParams['font.size'] = 12  # 字体大小
plt.rcParams['axes.unicode_minus'] = False  ## 设置正常显示符号

if __name__ == '__main__':
    data = np.genfromtxt('winequality_red.csv',
                         delimiter=';',
                         skip_header=True)
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # ols_lr = OLSLinearRegression()
    # ols_lr.train(X_train, y_train)
    # y_pred = ols_lr.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    # mae = mean_absolute_error(y_test, y_pred)
    # print("OSLinearRegression：")
    # print('mse=', mse)
    # print('mae=', mae)
    px = np.logspace(0, -8, 9)
    # 用于存储绘图数据
    pstep = []
    pmse = []
    for i in px:
        gd_lr = GDLinearRegression(n_iter=10 ** 10, eta=i, tol=0.00001)
        step = gd_lr.train(X_train, y_train)
        y_pred = gd_lr.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print("GDLinearRegression：")
        print('mse=', mse)
        print('mae=', mae)
        pstep.append(step)
        pmse.append(mse)
    # 作函数关系图
    fig, axis = plt.subplots(nrows=1, ncols=2, dpi=300)
    px = np.log10(px)
    pstep = np.log10(pstep)
    axis[0].plot(px, pstep, 'r-', lw=1, label='步长与收敛速度')
    axis[0].set_xlabel('lg(步长)')
    axis[0].set_ylabel('lg(收敛速度)')
    axis[0].set_title('步长与收敛速度的关系', fontdict={'size': 10})
    axis[0].legend(loc='best')
    axis[1].plot(px, pmse, 'b-', lw=1, label='步长与收敛误差')
    axis[1].set_xlabel('lg(步长)')
    axis[1].set_ylabel('收敛误差')
    axis[1].set_title('步长与收敛误差的关系', fontdict={'size': 10})
    axis[1].legend(loc='best')
    plt.savefig('步长对收敛速度和收敛误差的影响.png', dpi=1000)
    plt.show()
