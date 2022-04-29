from graphviz import Digraph

B = Digraph('G', filename='user.gv')
# B.node('a')
B.edge('a', '1')
B.edge('a', '2')
B.edge('a', '3')
B.edge('a', '5')
B.edge('a','f')
B.view()
