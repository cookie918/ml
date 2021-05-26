# -*- coding: UTF-8 -*-
# description   : 随机搜索

# 3. 随机搜索
# 使用随机搜索代替网格搜索的动机是，在许多情况下，所有的超参数可能不是同等重要的。
# 随机搜索从超参数空间中随机选择参数组合，参数由n_iter给定的固定迭代次数的情况下选择。
#
# 场景：
# 超参数组合比较多

# 缺点：
# 相对来讲稳定性差，可以相对比较好的超参数
#
# 优点：
# 效率高，实验证明，随机搜索的结果优于网格搜索。

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import KFold , cross_val_score
from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV

wine = load_wine()
X = wine.data
y = wine.target

#splitting the data into train and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 14)

knn = KNeighborsClassifier()

rand_param = { 'n_neighbors' : list(range(2,11)) ,
              'algorithm' : ['auto','ball_tree','kd_tree','brute'] }

rand_ser = RandomizedSearchCV(knn,rand_param,n_iter=10)
rand_ser.fit(X_train,y_train)

# best parameter combination
print('rand_ser.best_params_', rand_ser.best_params_)  # {'algorithm': 'auto', 'n_neighbors': 5}

# Score achieved with best parameter combination
print('rand_ser.best_score_', rand_ser.best_score_)  # 0.774

# all combinations of hyperparameters
print("rand_ser.cv_results_['params']", rand_ser.cv_results_['params'])

# average scores of cross-validation
print("rand_ser.cv_results_['mean_test_score']", rand_ser.cv_results_['mean_test_score'])
