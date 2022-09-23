import numpy as np


def extract(X, y, pos=1, neg=-1):
    """
    This function puts X and y into the new data,the new X and the new y, that the new y equals pos or neg regarding y iputed equals 2 or 5 and the new X is according the new y
    :param X:the feature of the data divided
    :param y:the label of the data divided
    :param pos:the positive label will output
    :param neg:the negative label will output
    :return:
    X: the new x,named X feather array
    y: the new y,named y label vector
    """
    idx = np.array(np.where(y == 2, 1, 0) + np.where(y == 5, 1, 0), dtype=bool)
    y = y[idx]
    y = np.where(y == 2, pos, neg)
    X = X[idx]
    print(f'X_25:\n{X}\n'
          f'y_25:\n{y}')
    return X, y


if __name__ == '__main__':
    pass
