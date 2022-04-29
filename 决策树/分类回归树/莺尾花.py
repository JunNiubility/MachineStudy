import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from decision_tree_cart import CartClassificationTree


def transform(fearture):
    le = LabelEncoder()
    le.fit(fearture)
    return le.transform(fearture)


def experiment(X_, y_):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    cct = CartClassificationTree()
    cct.train(X_train, y_train)
    y_predict = cct.predict(X_test)
    return accuracy_score(y_test, y_predict)


if __name__ == '__main__':
    dataurl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    X = np.genfromtxt(dataurl, delimiter=',', usecols=range(4), dtype=np.float_)
    y = np.genfromtxt(dataurl, delimiter=',', usecols=4, dtype=np.str_)
    le = LabelEncoder()
    y = transform(y)
    experiment(X, y)
    # np.mean([experiment(X,y) for _ in range(100)])
