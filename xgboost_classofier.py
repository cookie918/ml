import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets.samples_generator import make_classification
import xgboost as xgb

# 生成训练数据
X, y = make_classification(
    n_samples=500,
    n_features=20,
    n_classes=2,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# 构建模型
xgb_classifier = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.5,
    verbosity=1,
    objective='binary:logistic',
    random_state=1
)
# 训练模型
xgb_classifier.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="error",
        eval_set=[(X_test, y_test)])

# 网格搜索调参
xgb_classifierCv = GridSearchCV(xgb_classifier,
                   {'max_depth': [4,5,6],
                    'n_estimators': [5,10,20],
                    'learning_rate': [0.1,0.2,0.3,0.04,0.05,0.06,0.07]
                    }
                                )
xgb_classifierCv.fit(X_train,y_train)
print(xgb_classifierCv.best_score_)
print(xgb_classifierCv.best_params_)