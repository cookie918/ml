# -*- coding: UTF-8 -*-
# description   : 贝叶斯搜索
# 4. 贝叶斯搜索
# 贝叶斯优化属于一类优化算法，称为基于序列模型的优化(SMBO)算法。这些算法使用先前对损失f的观察结果，以确定下一个(最优)点来抽样f。该算法大致可以概括如下。
#
# 使用先前评估的点X1*:n*，计算损失f的后验期望。
# 在新的点X的抽样损失f，从而最大化f的期望的某些方法。该方法指定f域的哪些区域最适于抽样。
# 重复这些步骤，直到满足某些收敛准则。

# 缺点：
#
# 要在2维或3维的搜索空间中得到一个好的代理曲面需要十几个样本，增加搜索空间的维数需要更多的样本。

import warnings

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from skopt import BayesSearchCV

warnings.filterwarnings("ignore")

# parameter ranges are specified by one of below

wine = load_wine()
X = wine.data
y = wine.target

# splitting the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)

knn = KNeighborsClassifier()
# defining hyper-parameter grid
grid_param = {'n_neighbors': list(range(2, 11)),
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

# initializing Bayesian Search
Bayes = BayesSearchCV(knn, grid_param, n_iter=30, random_state=14)
Bayes.fit(X_train, y_train)

# best parameter combination
print('Bayes.best_params_', Bayes.best_params_)  # {'algorithm': 'auto', 'n_neighbors': 5}

# Score achieved with best parameter combination
print('Bayes.best_score_', Bayes.best_score_)  # 0.774

# all combinations of hyperparameters
print("Bayes.cv_results_['params']", Bayes.cv_results_['params'])

# average scores of cross-validation
print("Bayes.cv_results_['mean_test_score']", Bayes.cv_results_['mean_test_score'])
