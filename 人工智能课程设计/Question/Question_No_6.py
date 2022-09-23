import Question_No_2 as qN2
import Question_No_5 as qN5

if __name__ == '__main__':
    X_train, y_train = qN2.Data_prepare('tra')
    X_train25, y_train25 = qN5.extract(X_train, y_train)
    print(f'X_train25.shape:{X_train25.shape}\n'
          f'y_train25.shape:{y_train25.shape}')
