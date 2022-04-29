import numpy as np
from svm import SMO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际c序号对uda应）
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  #（代表仅使用第0号GPU）

if __name__ == "__main__":
    dataurl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
    X = np.genfromtxt(dataurl, delimiter=',', usecols=range(1, 17))
    y = np.genfromtxt(dataurl, delimiter=',', usecols=[0], dtype=np.str_)
    y = np.where(y == 'C', 1, -1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    acc_list = []
    p_list = []
    C_list = np.logspace(-2, 1, 4)
    gamma_list = np.logspace(-2, 1, 4)
    for C in C_list:
        for gamma in gamma_list:
            print((C, gamma))
            clf = SMO(C=C, tol=0.01, kernel='rbf', gamma=gamma)
            clf.train(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            acc_list.append(accuracy)
            p_list.append((C, gamma))
    idx = np.argmax(acc_list)
    print('最优超参数：')
    print(p_list[idx])
    print(f'best accuracy:{acc_list[idx]}')
