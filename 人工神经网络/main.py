import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from ann_classification import ANNClassifier
import matplotlib.pyplot as plt
import os

# 创建fig文件夹
try:
    os.mkdir('fig')
except OSError:
    pass

if __name__ == "__main__":
    # 导入并分割数据
    tra_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra'
    tes_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes'
    data_train = np.genfromtxt(tra_url, delimiter=',', dtype=float)
    data_test = np.genfromtxt(tes_url, delimiter=',', dtype=float)
    # 使用切片获取训练和测试数据
    X_train, y_train = data_train[:, :-1], data_train[:, -1]
    X_test, y_test = data_test[:, :-1], data_test[:, -1]

    # 标准化
    ss = StandardScaler(copy=True, with_mean=True, with_std=True)
    ss.fit(X_train)
    X_train_std = ss.transform(X_train)
    ss.fit(X_test)
    X_test_std = ss.transform(X_test)

    # 针对y_train、y_test完成one-hot编码（即1 of n）编码
    lb = LabelBinarizer()
    y_train_bin = lb.fit_transform(y_train)
    # etalist = [0.1, 0.3, 0.5]
    # for eta in etalist:
    #     clf = ANNClassifier(hidden_layer_sizes=(10,), eta=eta, max_iter=500, tol=0.00001)
    #     clf.train(X_train_std, y_train_bin)
    #     y_pred_bin = clf.predict(X_test_std)
    #     y_pred = lb.inverse_transform(y_pred_bin)
    #     accuracy = accuracy_score(y_test, y_pred)
    #     print(f'eta={eta},accuracy={accuracy}')
    #
    # # 将tra[0:9]的数据转为方阵图，保存并显示
    # for i in range(20):
    #     tra1 = data_train[i, :-1]
    #     tra2 = tra1.reshape(8, 8)
    #     plt.matshow(tra2,cmap=plt.cm.binary)
    #     plt.savefig('fig/' + str(i) + '.jpg')
