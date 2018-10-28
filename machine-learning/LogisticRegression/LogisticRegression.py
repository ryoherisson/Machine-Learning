import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 学習データの読み込み
train = pd.read_csv('train.csv')

# data clearning
# かけているデータの箇所を埋める
# 乗船クラスから年齢を予想して埋める
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)

# cabin列を削除
train.drop('Cabin', axis=1, inplace=True)

# 欠損値がある行を削除
train.dropna(inplace=True)

# ダミー変数を作成
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

# ダミー変数を加えたデータを作成
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)


# データを学習用とテスト用に分解
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),
                                                    train['Survived'], test_size=0.30,
                                                    random_state=0)
# 学習
lr = LogisticRegression()
lr.fit(X_train, y_train)

# 予測
pred = lr.predict(X_test)
print(classification_report(y_test,pred))
