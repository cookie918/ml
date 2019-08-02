import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# 读取数据
train = pd.read_csv('./dataset/train_modified.csv')[:500]

# 数据摸底
print(train['Disbursed'].value_counts())

# 构建训练数据
target = 'Disbursed'
IDcol = 'ID'
X_columns = [ x for x in train.columns if x not in [target, IDcol]]
X = train[X_columns]
y = train[target]
# 构建模型
gbdt = GradientBoostingClassifier()
# 训练模型
gbdt.fit(X, y)

# 预测
print(gbdt.predict(X[:2]))
# 评价
print(gbdt.predict_proba(X[:2]))

