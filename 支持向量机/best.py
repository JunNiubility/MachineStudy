import numpy as np
from svm import SMO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    dataurl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
    X = np.genfromtxt(dataurl, delimiter=',', usecols=range(1, 17))
    y = np.genfromtxt(dataurl, delimiter=',', usecols=[0], dtype=np.str_)
    y = np.where(y == 'C', -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = SMO(C=5, tol=0.01, kernel='rbf', gamma=0.05)
    clf.train(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    C = confusion_matrix(y_test, y_pred)
    print(f'accuracy={accuracy}\nC={C}')
