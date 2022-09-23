import Question_No_2 as qN2
import Question_No_5 as qN5
import matplotlib.pyplot as plt

import os

# make png dir
try:
    os.mkdir('png')
except OSError:
    pass


def save_picture(X, y, idx=None, pos=1):
    """
    This function saves as a picture from X feather array
    :param X:X feather array
    :param y:y label vector
    :param idx:
    :param pos:the index of X feather array or y label vector
    :return:None
    """
    if idx is None:
        idx = [0, ]
    for item in idx:
        picture = X[item].reshape(8, 8)
        plt.matshow(picture, cmap=plt.cm.gray)
        plt.axis('off')
        plt.savefig('png/' + ('p_' if y[item] == pos else 'n_') + str(item) + '.png')
        if __name__ == '__main__':
            plt.show()


if __name__ == '__main__':
    X_train, y_train = qN2.Data_prepare('tra')
    X_train25, y_train25 = qN5.extract(X_train, y_train)
    save_picture(X_train25, y_train25)
