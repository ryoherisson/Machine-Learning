import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pokemon = pd.read_csv('https://raw.githubusercontent.com/we-b/datasets_for_ai/master/poke.csv')

X = np.array(pokemon)
X = X[:, 2:8]

# PCAで学習
pca = PCA(n_components=2)
pca.fit()

# 学習したPCAで主成分を抽出
X_pca = pca.transform(X)
print('主成分を抽出')
print(X_pca)

# 散布図に描画
x = X_pca[:, 0]
y = X_pca[:, 1]
fig, ax = plt.subplots(figsize=(8,8))
plt.scatter(x, y)
plt.xlabel('1st principal component')
plt.ylabel('2nd principal component')
plt.show()

# 寄与率
print('寄与率')
print(pca.explained_variance_ratio_)
print('因果負荷量')
print(pca.components_)

# 散布図に描画
attr = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

fig, ax = plt.subplots(figsize=(8, 8))
plt.scatter(x, y)
for i in range(pca.components_.shape[1]):
    x1 = pca.components_[0, i]*100
    y1 = pca.components_[1, i]*100
    ax.arrow(0, 0, x1, y1, head_width=5, head_length=10, fc='k', ec='k')
    plt.text(x1+15, y1, attr[i])
plt.showO()
