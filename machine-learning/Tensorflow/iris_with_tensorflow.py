import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report,confusion_matrix


# データ読み込み
df = pd.read_csv('iris.csv')
# 列ごとのタイトル
df.columns = ['sepal_length','sepal_width','petal_length','petal_width','target']

# X,yに代入
X = df.drop('target',axis=1)
y = df['target'].apply(int)

# 学習用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


feat_cols = []

for col in X.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))

# 予測モデル
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=5,shuffle=True)
classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3,feature_columns=feat_cols)
classifier.train(input_fn=input_func,steps=50)

pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
note_predictions = list(classifier.predict(input_fn=pred_fn))

final_preds  = []
for pred in note_predictions:
    final_preds.append(pred['class_ids'][0])

print(confusion_matrix(y_test,final_preds))
print(classification_report(y_test,final_preds))
