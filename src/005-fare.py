#!/usr/local/bin/python3

# use age for lineaer regression
# accuracy 0.7928
# kaggle score 0.7703

import os
import sys  # pylint: disable=unused-import

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath("../../utils"))
from pjo_plot import plot_histograms  # pylint: disable=import-error,wrong-import-position,unused-import

warnings.simplefilter(action='ignore', category=RuntimeWarning)
pd.options.display.float_format = '{:.4f}'.format

# load data
train = pd.read_csv('../data/train.csv')
test = pd.read_csv("../data/test.csv")


# fill missing values with mean
meanAge = train.Age.mean()
train.Age.fillna(meanAge, inplace=True)
test.Age.fillna(meanAge, inplace=True)

train.Fare.fillna(train.Fare.mean(), inplace=True)
test.Fare.fillna(test.Fare.mean(), inplace=True)

# train.Fare = np.log(1 + train.Fare)
# test.Fare = np.log(1 + test.Fare)

# print(train.describe())


# plot_histograms(train)

# look for outliers
# plt.scatter(range(len(train.Age)), train.Age)
# plt.show()
# or
# plt.boxplot(train.Age)
# plt.show()

#-------- main

# feature creation
train['is_female'] = train.Sex.apply(lambda sex: 1 if sex == 'female' else 0)
test['is_female'] = test.Sex.apply(lambda sex: 1 if sex == 'female' else 0)
# print(train.describe())

# print(train.describe())

# remove outliers
quartile_high, quartile_low = np.percentile(train.Age, [75, 25])
inter_quartile_range = quartile_high - quartile_low

train = train[(train.Age >= quartile_low - inter_quartile_range * 1.5) &
              (train.Age < quartile_high + inter_quartile_range * 1.5)]

# plt.scatter(range(len(train.Age)), train.Age)
# plt.show()

train = train[['Fare', 'Age', 'is_female', 'Pclass', 'Survived']]

x_train = train.drop('Survived', axis=1)
y_train = train.Survived
x_test = test[x_train.columns]

# print(x_train.head(20))

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
print('submission saved')
