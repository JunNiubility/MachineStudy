import numpy as np
from sklearn.preprocessing import LabelEncoder
from decision_tree import DecisionTree
from decision_tree_plotter import DecisionTreePloter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# convert函数: 对每一列进行转换
def convert(col, value_name_list):
    le = LabelEncoder()
    res = le.fit_transform(col)
    value_name_list.append(le.classes_)
    return res


if __name__ == "__main__":
    dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
    dataset = np.genfromtxt(dataset_url, delimiter=',', dtype=np.str_)
    le = LabelEncoder()
    col = dataset[:, 0]
    le.fit(col)
    le.transform(col)

    value_name_list = []
    dataset = np.apply_along_axis(convert, axis=0, arr=dataset, value_name_list=value_name_list)
    X = dataset[:, :-1]
    y = dataset[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    dt = DecisionTree()
    dt.train(X_train, y_train)

    feature_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

    feature_dict = {
        i: {'name': v,
            'value_names': dict(enumerate(value_name_list[i]))}
        for i, v in enumerate(feature_names)
    }

    label_dict = dict(enumerate(value_name_list[-1]))

    plotter = DecisionTreePloter(dt.tree_, feature_names=feature_dict, label_names=label_dict)
    plotter.plot()
    y_predict = dt.predict(X_test)
    score = accuracy_score(y_test, y_predict)
    print(f'accuracy_score:{score}')
