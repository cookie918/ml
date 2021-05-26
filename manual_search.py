
# -*- coding: UTF-8 -*-
# description   : 传统手工搜索
# 1. 传统手工搜索
# 在传统的调参过程中，我们通过训练算法手动检查随机超参数集，并选择符合我们目标的最佳参数集。
#
# 我们看看代码：
# 缺点：
#
# 没办法确保得到最佳的参数组合。
# 这是一个不断试错的过程，所以，非常的耗时。

#importing required libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold , cross_val_score
from sklearn.datasets import load_wine

wine = load_wine()
X = wine.data
y = wine.target

#splitting the data into train and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 14)

#declaring parameters grid
k_value = list(range(2,11))
algorithm = ['auto','ball_tree','kd_tree','brute']
scores = []
best_comb = []
kfold = KFold(n_splits=5)
index=0

#hyperparameter tunning
for algo in algorithm:
  for k in k_value:
    knn = KNeighborsClassifier(n_neighbors=k,algorithm=algo)
    results = cross_val_score(knn,X_train,y_train,cv = kfold)
    index = index + 1
    print(f'{index}_Score:{round(results.mean(),4)} with algo = {algo} , K = {k}')
    scores.append(results.mean())
    best_comb.append((k,algo))

best_param = best_comb[scores.index(max(scores))]
print(f'\nThe Best Score : {max(scores)}')
print(f"['algorithm': {best_param[1]} ,'n_neighbors': {best_param[0]}]")
