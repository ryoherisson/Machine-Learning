import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# csvファイルを読み込み
USAHousing = pd.read_csv('USA_Housing.csv')

# 特徴量と目的変数を定義
X = USAHousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAHousing['Price']

# データを学習用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# トレーニングモデルの作成
lr = LinearRegression()
lr.fit(X_train, y_train)

# 予測
pred = lr.predict(X_test)

# 散布図
plt.scatter(y_test, pred)
plt.show()


print('MAE: ', metrics.mean_absolute_error(y_test, pred))
print('MSE: ', metrics.mean_squared_error(y_test, pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, pred)))
