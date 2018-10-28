import matplotlib
matplotlib.use('TkAgg') #グラフが描画されないので念のため記載
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# データの読み込み
iris = sns.load_dataset('iris')


# 学習用データとテスト用データに分割
X = iris.drop(['species'], axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 学習
svc_model = SVC()
svc_model.fit(X_train, y_train)

# 予測
pred = svc_model.predict(X_test)

# モデルの評価
print('confusion matrix')
print(confusion_matrix(y_test, pred))
print('\n')
print('classification report')
print(classification_report(y_test, pred))

# GridSearch
params_grid = {'C':[0.1, 1.0, 10, 100], 'gamma':[1, 0.1, 0.01, 0.001]}
grid = GridSearchCV(SVC(), params_grid, refit='True', verbose=2)
grid.fit(X_train, y_train)

grid_pred = grid.predict(X_test)
print('confusion matrix')
print(confusion_matrix(y_test, grid_pred))
print('\n')
print('classification report')
print(classification_report(y_test, grid_pred))
