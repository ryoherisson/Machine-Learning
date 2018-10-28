import matplotlib
matplotlib.use('TkAgg') #グラフが描画されないので念のため記載
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ダミーデータ作成
data = make_blobs(n_samples=300, n_features=2, centers=4, cluster_std=1.8,random_state=10)

# 元データをプロット
plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='rainbow')
plt.show()

# クラスター４つのモデル作成
kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])

# グラフを描画
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
plt.show()
