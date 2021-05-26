# -*- coding: UTF-8 -*-
# description   : 网格搜索（暴力搜索）
#2. 网格搜索（暴力搜索）
#网格搜索是一种基本的超参数调优技术。它类似于手动调优，为网格中指定的所有给定超参数值的每个排列构建模型，评估并选择最佳模型。
# 考虑上面的例子，其中两个超参数k_value =[2,3,4,5,6,7,8,9,10] & algorithm =[' auto '， ' ball_tree '， ' kd_tree '， ' brute ']，
# 在这个例子中，它总共构建了9*4 = 36不同的模型。
# 至于为什么二者的结果会不一样，那是因为seed种子数也是一个超参数
#
# 场景：
# 因此它只适用于超参数数量小的情况。

# 缺点：
# 效率低下
#
# 优点：
# 相对来讲稳定性，并且可以找到比较好的超参数

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold , cross_val_score
from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV

wine = load_wine()
X = wine.data
y = wine.target

#splitting the data into train and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 14)

knn = KNeighborsClassifier()
grid_param = {'n_neighbors': list(range(2, 11)),
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid = GridSearchCV(knn, grid_param, cv=5)
grid.fit(X_train, y_train)

# best parameter combination
print('grid.best_params_', grid.best_params_)  # {'algorithm': 'auto', 'n_neighbors': 5}

# Score achieved with best parameter combination
print('grid.best_score_', grid.best_score_)  # 0.774

# all combinations of hyperparameters
print("grid.cv_results_['params']", grid.cv_results_['params'])

# average scores of cross-validation
print("grid.cv_results_['mean_test_score']", grid.cv_results_['mean_test_score'])


