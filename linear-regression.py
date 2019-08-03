import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.linear_model import LinearRegression

# 构建训练数据
data = pd.read_excel('./dataset/Folds5x2_pp.xlsx')
X_columns = data.columns[:4]
y_columns = data.columns[4:]
print(X_columns)
print(y_columns)
data_X, data_y = data[X_columns], data[y_columns]
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# 构建模型
linreg = LinearRegression()
# 训练模型
linreg.fit(X_train, y_train)
# 预测
y_pred = linreg.predict(X_test)
# 模型评价
print('MSE== ', metrics.mean_squared_error(y_test, y_pred))
print('RMSE== ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# 交叉验证
y_cross_pred = cross_val_predict(linreg, X_train, y_train, cv=10)
print('MSE== ', metrics.mean_squared_error(y_train, y_cross_pred))
print('RMSE== ', np.sqrt(metrics.mean_squared_error(y_train, y_cross_pred)))
