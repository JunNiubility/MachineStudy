import numpy as np
from sklearn.preprocessing import StandardScaler


def Data_prepare(data_url='tra'):
    """
    This function prepares the data needed
    :param data_url: the last name of the data file,you can fill "tra" or "tes" in
    :return:None
    """
    data = np.genfromtxt(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.' + data_url,
        delimiter=',', dtype=float)
    X_train, y_train = data[:, :-1], data[:, -1]

    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)

    if __name__ == '__main__':
        print(f'data:\n{data}\n'
              f'X_train:\n{X_train}\n'
              f'y_train:\n{y_train}')


if __name__ == '__main__':
    Data_prepare()
