import Question_No_2 as qN2
import Question_No_3 as qN3

if __name__ == '__main__':
    X_train, _ = qN2.Data_prepare()
    idx = qN3.search2and5()
    X_train25 = X_train[idx]
    print(f'X_train25:\n{X_train25}')
