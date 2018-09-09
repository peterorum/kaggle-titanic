#!/usr/local/bin/python3

# use xgboost
# accuracy 0.87
# kaggle score 0.7655

import os
import sys  # pylint: disable=unused-import

import warnings
import matplotlib.pyplot as plt  # pylint: disable=unused-import
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath("../../utils"))
from pjo_plot import plot_histograms  # pylint: disable=import-error,wrong-import-position,unused-import

warnings.simplefilter(action='ignore')
pd.options.display.float_format = '{:.4f}'.format

# load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv("../input/test.csv")


# fill missing values with mean
meanAge = train.Age.mean()
train.Age.fillna(meanAge, inplace=True)
test.Age.fillna(meanAge, inplace=True)

train.Fare.fillna(train.Fare.mean(), inplace=True)
test.Fare.fillna(test.Fare.mean(), inplace=True)

#-------- main

# feature creation
train['is_female'] = train.Sex.apply(lambda sex: 1 if sex == 'female' else 0)
test['is_female'] = test.Sex.apply(lambda sex: 1 if sex == 'female' else 0)

train = train[['Fare', 'Age', 'is_female', 'Pclass', 'Survived']]

x_train = train.drop('Survived', axis=1)
y_train = train.Survived
x_test = test[x_train.columns]

reg = xgb.XGBRegressor()
reg.fit(x_train, y_train)

reg_train_pred = reg.predict(x_train)

reg_train_pred = np.round(reg_train_pred).astype(int)

print('accuracy', accuracy_score(y_train, reg_train_pred))

reg_test_pred = reg.predict(x_test)
reg_test_pred = np.round(reg_test_pred).astype(int)

predicted = pd.DataFrame({
    "PassengerId": test.PassengerId,
    "Survived": reg_test_pred
})

predicted.to_csv('../input/submission.csv', index=False)
print('submission saved')
