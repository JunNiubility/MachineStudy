import numpy as np
from decision_tree import DecisionTree
from decision_tree_plotter import DecisionTreePloter

if __name__=="__main__":
    dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/lenses/lenses.data'
    dataset = np.genfromtxt(dataset_url, dtype=np.int_)
    X = dataset[:, 1:-1]
    y = dataset[:, -1]
    dt = DecisionTree()
    dt.train(X, y)

    fractions_dict = {
        0: {'name': 'age',
            'value_names': {1: 'yang',
                            2: 'pre-presbyopic',
                            3: 'presbyopic'}
            },
        1: {'name': 'prescript',
            'value_names': {1: 'myope',
                            2: 'hypermetrope'}
            },
        2: {'name': 'astigmatic',
            'value_names': {1: 'no',
                            2: 'yes'}
            },
        3: {'name': 'tear rate',
            'value_names': {1: 'reduced',
                            2: 'normal'}
            }
    }

    label_dict = {
        1: 'head',
        2: 'soft',
        3: 'no_lenses',
    }

    dtp = DecisionTreePloter(dt.tree_,
                             feature_names=fractions_dict,
                             label_names=label_dict)
    dtp.plot()
