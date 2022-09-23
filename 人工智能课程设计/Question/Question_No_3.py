import Question_No_2 as qN2
import numpy as np


def search2and5(data_url='tra'):
    """
    This function searchsfor whether the label is 2 or 5 in the data，and find those index
    :param data_url: the last name of the data file,you can fill "tra" or "tes" in
    :return:
    idx: the index  have been found by this function
    """
    X_train, y_train = qN2.Data_prepare(data_url)
    idx = np.array(np.where(y_train == 2, 1, 0) + np.where(y_train == 5, 1, 0), dtype=bool)
    y_train = y_train[idx]
    y_train25 = np.where(y_train == 2, 1, -1)
    if __name__ == '__main__':
        print(f'y_equal2or5:\n{y_train}\n'
              f'y_train25:\n{y_train25}')
    return idx


if __name__ == '__main__':
    search2and5()
