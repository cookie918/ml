import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_gaussian_quantiles

# 生成2维正态分布，生成的数据按分位数分为两类，500个样本,2个样本特征，协方差系数为2
X1, y1 = make_gaussian_quantiles(cov=2.0,n_samples=500, n_features=2,n_classes=2, random_state=1)
print(X1, y1)
# 生成2维正态分布，生成的数据按分位数分为两类，400个样本,2个样本特征均值都为3，协方差系数为2
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,n_samples=500, n_features=2, n_classes=2, random_state=1)
print(X2, y2)

#构建训练数据
X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1))
print(X, y)

# 构建模型
bdt = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
    algorithm="SAMME",
    n_estimators=200,
    learning_rate=0.8)

# 训练模型
bdt.fit(X, y)

# 模型准确率

print(bdt.score(X, y))






