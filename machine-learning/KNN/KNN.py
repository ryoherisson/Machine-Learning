import matplotlib
matplotlib.use('TkAgg') #グラフが描画されないので念のため記載
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# データの読み込み
df = pd.read_csv('Classified Data', index_col=0)

# データの標準化
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])


# 学習用データとテスト用データに分割
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['TARGET CLASS'], test_size=0.3)

# 学習
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# 予測
pred = knn.predict(X_test)

# モデルの評価
print('K=23')
print('confusion matrix')
print(confusion_matrix(y_test, pred))
print('\n')
print('classification report')
print(classification_report(y_test, pred))
print('\n')


# Kの値の選択
error_rate=[]
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# 横軸kの値、縦軸error_rateエラーグラフを描画
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs K.Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()


# グラフの結果からK=23に設定
knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

# モデルの評価
print('K=23')
print('confusion matrix')
print(confusion_matrix(y_test, pred))
print('\n')
print('classification report')
print(classification_report(y_test, pred))
