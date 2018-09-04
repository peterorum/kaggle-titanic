#!/usr/local/bin/python3

# use age for lineaer regression
# accuracy 0.7890
# kaggle score 0.7655 (same as female alone)

import sys  # pylint: disable=unused-import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import warnings

warnings.simplefilter(action='ignore', category=RuntimeWarning)

# load data
train = pd.read_csv('../data/train.csv')
test = pd.read_csv("../data/test.csv")

#-------- main

# fill missing values
meanAge = train.Age.mean()
train.Age.fillna(meanAge, inplace=True)
test.Age.fillna(meanAge, inplace=True)

# feature creation
train['is_female'] = train.Sex.apply(lambda sex: 1 if sex == 'female' else 0)
test['is_female'] = test.Sex.apply(lambda sex: 1 if sex == 'female' else 0)
# print(train.describe())

train = train[['Age', 'is_female', 'Pclass', 'Survived']]

x_train = train.drop('Survived', axis=1)
y_train = train.Survived
x_test = test[x_train.columns]

print(x_train.head(20))

lr = LinearRegression()
lr.fit(x_train, y_train)

lr_train_pred = lr.predict(x_train)
lr_train_pred = np.round(lr_train_pred).astype(int)

print('accuracy', accuracy_score(y_train, lr_train_pred))

lr_test_pred = lr.predict(x_test)
lr_test_pred = np.round(lr_test_pred).astype(int)

predicted = pd.DataFrame({
    "PassengerId": test.PassengerId,
    "Survived": lr_test_pred
})

predicted.to_csv('../data/submission.csv', index=False)
