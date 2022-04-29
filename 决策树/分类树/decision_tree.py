import numpy as np


class DecisionTree:
    class Node:
        def __init__(self):
            self.value = None

            # 内部叶节点属性
            self.feature_index = None
            self.children = {}

        def __str__(self):
            if self.children:
                s = '内部节点<%s>:\n' % self.feature_index
                for fv, node in self.children.items():
                    ss = '[%s]->%s' % (fv, node)
                    s += '\t' + ss.replace('\n', '\n\t') + '\n'
            else:
                s = '叶节点(%s)' % self.value
            return s

    def __init__(self, gain_threshold=1e-2):
        # 信息增益阈值
        self.tree_ = None
        self.gain_threshold = gain_threshold

    @staticmethod
    def _entropy(y):
        """熵:-sum(pi*log(pi))"""
        c = np.bincount(y)
        p = c[np.nonzero(c)] / y.size
        return -np.sum(p * np.log2(p))

    def _conditional_entropy(self, feature, y):
        """条件熵"""
        feature_value = np.unique(feature)
        h = 0.
        for v in feature_value:
            y_sub = y[feature == v]
            p = y_sub.size / y.size
            h += p * self._entropy(y_sub)
        return h

    def _information_gain(self, feature, y):
        """信息增益=经验熵-经验条件熵"""
        return self._entropy(y) - self._conditional_entropy(feature, y)

    def _select_feature(self, X, y, feature_list):
        """选择信息增益最大的特征"""
        # 正常情况下，返回特征(最大信息增益)在feature_list中的index值
        if feature_list:
            gains = np.apply_along_axis(self._information_gain, 0, X[:, feature_list], y)
            index = np.argmax(gains)
            if gains[index] > self.gain_threshold:
                return index
        # 当feature_list已为空，或所有特征信息增益都小于阈值，返回None
        return None

    def _build_tree(self, X, y, feature_list):
        """决策树构造算法(递归)"""
        # 创建节点
        node = DecisionTree.Node()
        # 统计数据集汇总样本类标记的个数
        labels_count = np.bincount(y)
        # 任何情况下节点值总等于数据集中样本最多的类标记
        node.value = np.argmax(labels_count)

        # 判断类标记是否全部一致
        if np.count_nonzero(labels_count) != 1:
            # 选择信息增益最大的特征
            index = self._select_feature(X, y, feature_list)
            # 能选择到合适的特征时，创建内部节点，否则创建叶节点
            if index is not None:
                # 将已选特征从特征集合中删除
                node.feature_index = feature_list.pop(index)
                # 根据已选特征的取值划分数据集，并使用数据子集创建子树
                feature_values = np.unique(X[:, node.feature_index])
                for v in feature_values:
                    # 筛选出数据子集
                    idx = X[:, node.feature_index] == v
                    X_sub, y_sub = X[idx], y[idx]
                    # 创建子树
                    node.children[v] = self._build_tree(X_sub, y_sub,
                                                        feature_list.copy())
        return node

    def _predict_one(self, x):
        """搜索决策树，对单个实例进行预测"""
        # 爬树一直爬到某叶节点为止，返回叶节点的值
        node = self.tree_
        while node.children:
            child = node.children.get(x[node.feature_index])
            if not child:
                # 根据测试点属性值不能找到相应子树(这是有可能的)
                # 则停止搜索，将该内部节点当作叶节点(返回其值)
                node = child
                break
            return node.value

    def train(self, X_train, y_train):
        """训练"""
        _, n = X_train.shape
        self.tree_ = self._build_tree(X_train, y_train, list(range(n)))

    def predict(self, X):
        """预测"""
        # 对每一个实例调用_predict_ones,返回收集到的结果数组
        return np.apply_along_axis(self._predict_one, axis=1, arr=X)

    def __str__(self):
        """生成决策树的对应字符串(用于打印输出决策树)"""
        if hasattr(self, 'tree_'):
            return str(self.tree_)
        return ''
