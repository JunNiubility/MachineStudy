import Question_No_2 as qN2
import Question_No_5 as qN5

if __name__ == '__main__':
    X_test, y_test = qN2.Data_prepare('tes')
    X_test25, y_test25 = qN5.extract(X_test, y_test)
    print(f'X_test25.shape:{X_test25.shape}\n'
          f'y_test25.shape:{y_test25.shape}')
