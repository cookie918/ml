from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


# 仍然使用自带的iris数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 训练模型，树模型最大的深度是４
decision_tree_classifier = DecisionTreeClassifier(max_depth=4)

# 训练模型
decision_tree_classifier.fit(X, y)

# 可视化决策树

from sklearn import  tree
import pydotplus

dot_data = tree.export_graphviz(decision_tree_classifier, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")