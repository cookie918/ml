import  numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# 训练数据准备
iris = datasets.load_iris()
print(iris.target_names)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.30, random_state=0)

# 选择模型
cls = LogisticRegression()

# 把数据交给模型训练
cls.fit(X_train, y_train)

print(cls.score(X_test, y_test))