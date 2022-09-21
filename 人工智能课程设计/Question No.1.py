import numpy as np
from sklearn.preprocessing import StandardScaler


def Data_prepare(data_url='tra'):
    data = np.genfromtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.' + data_url,
                         delimiter=',', dtype=float)
    X_train, y_train = data[:, :-1], data[:, -1]
    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)
    print(f'data:\n{data}\nX_train:\n{X_train}\ny_train:\n{y_train}')


if __name__ == '__main__':
    Data_prepare()
