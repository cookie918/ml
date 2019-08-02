import numpy as np
from sklearn.naive_bayes import GaussianNB

# 训练数据集
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2],[1, 2]])
y = np.array([1, 1, 1, 2, 2, 2, 3])

# 构建模型
clf = GaussianNB()

# 训练模型
clf.fit(X, y)

# 模型预测
print('== 模型预测类型　＝＝')
print(clf.predict([[-0.8, -1]]))

print('== 预测每种类别概率 ==')
print(clf.predict_proba([[-0.8, -1]]))