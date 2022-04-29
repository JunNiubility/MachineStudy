from graphviz import Digraph


class DecisionTreePloter:
    def __init__(self, tree, feature_names=None, label_names=None):
        # 保存决策树
        self.tree = tree
        # 保存特征名称字典
        self.feature_names = feature_names
        # 保存类标记名字字典
        self.label_names = label_names
        # 创建图(graphviz)
        self.graph = Digraph("Decision Tree")

    def _build(self, dt_node):
        # 根据决策树中的节点，创建graphviz图中一个节点
        if dt_node.children:
            # dt_node是内部节点
            # 获取特征名字
            d = self.feature_names[dt_node.feature_index]
            if self.feature_names:
                label = d['name']
            else:
                label = str(dt_node.feature_index)
            # 创建方形内部节点(graphviz)
            self.graph.node(str(id(dt_node)), label=label, shape='')
            for feature_value, dt_child in dt_node.children.items():
                # 递归调用_build创建子节点(graphviz)
                self._build(dt_child)
                # 获得特征值的名字
                d_value = d.get('value_names')
                if d_value:
                    label = d_value[feature_value]
                else:
                    label = str(feature_value)
                # 创建连接父子节点的边(graphviz)
                self.graph.edge(str(id(dt_node)), str(id(dt_child)), label=label)
        else:
            # dt_node是叶节点
            # 获取类标记的名字
            if self.label_names:
                label = self.label_names[dt_node.value]
            else:
                label = str(dt_node.value)
            # 创建圆形叶子节点(graphviz)
            self.graph.node(str(id(dt_node)), label=label, shape='')

    def plot(self):
        # 创建graphviz图
        self._build(self.tree)
        # 显示图
        self.graph.view()
