import numpy as np
from sklearn.preprocessing import StandardScaler


def Data_prepare(data_url='tra'):
    """
    This function prepares the data used in the train model
    :param data_url:the last name of the data file,you can fill "tra" or "tes" in
    :return :
    X:the parameter of a piece of data
    y:the label of a piece of data
    """
    data = np.genfromtxt(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.' + data_url,
        delimiter=',', dtype=float)
    X, y = data[:, :-1], data[:, -1]

    ss = StandardScaler()
    ss.fit(X)
    X = ss.transform(X)

    if __name__ == '__main__':
        print(f'the shape of X of {data_url} is {np.shape(X)}\n'
              f'the shape of y of {data_url} is {np.shape(y)}')
    return X, y


if __name__ == '__main__':
    Data_prepare('tra')
    Data_prepare('tes')
