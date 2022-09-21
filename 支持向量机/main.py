from svm import *
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    # C_list = np.logspace(-1, 1, num=5)
    # gamma_list = np.logspace(-3, -1, num=5)
    X_train, X_test, y_train, y_test = Data_prapare()
    acc_list, par_list, y_pred = Search_best(X_train, X_test, y_train, y_test)

    idx = np.argmax(acc_list)
    precison = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    confusionmatrix = confusion_matrix(y_test, y_pred)

    # print('最优超参数：')
    print('训练结果评价：')
    # print(par_list[idx])
    # print(f'best accuracy:{acc_list[idx]}')
    print(f'confusion matrix:\n{confusion_matrix(y_test, y_pred)}')
    print(f'accuracy:{acc_list[idx]}')
    print(f'precison:{precision_score(y_test, y_pred)}')
    print(f'recall:{recall_score(y_test, y_pred)}')
