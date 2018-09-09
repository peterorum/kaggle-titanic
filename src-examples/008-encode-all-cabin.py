# encode all. add cabin floor.
# accuracy 0.8810
# kaggle score 0.7799

import os
import sys  # pylint: disable=unused-import

import warnings

import category_encoders as ce
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
mean_age = train.Age.mean()
train.Age.fillna(mean_age, inplace=True)
test.Age.fillna(mean_age, inplace=True)

train.Fare.fillna(train.Fare.mean(), inplace=True)
test.Fare.fillna(test.Fare.mean(), inplace=True)

# first letter if a string


def get_level(cabin):
    level = '0'

    if (type(cabin) is str):
        if len(cabin) >= 1:
            level = cabin[0]

    return level

# first letter of cabin
train['level'] = train.Cabin.apply(lambda c: get_level(c))
test['level'] = test.Cabin.apply(lambda c: get_level(c))

# select columns we can deal with
train = train[['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'level']]

categorical_cols = [col for col in train.columns if train[col].dtype == 'object']

numeric_cols = [col for col in train.columns if (train[col].dtype == 'int64') | (train[col].dtype == 'float64')]

train_cols = numeric_cols + categorical_cols

key = 'PassengerId'
target = 'Survived'

# in case not numeric
if key in categorical_cols:
    categorical_cols.remove(key)

# for encoding merge, add key
if key not in train_cols:
    train_cols.append(key)

# add target to retain it on separating
if target not in train_cols:
    train_cols.append(target)

# just train cols
train = train[train_cols]

# test uses same cols except target
test_cols = train_cols.copy()
test_cols.remove(target)

test = test[test_cols]

# need to combine to encode
all_data = pd.concat([train, test])

onehot_encoder = ce.OneHotEncoder(cols=categorical_cols)
all_data = onehot_encoder.fit_transform(all_data)

# separate again
train = all_data[all_data[key].isin(train[key])].copy()
test = all_data[all_data[key].isin(test[key])].copy()

#-------- main

# create train & test data
x_train = train.drop([key, target], axis=1)
y_train = train[target]
x_test = test[x_train.columns]

reg = xgb.XGBRegressor()
reg.fit(x_train, y_train)

reg_train_pred = reg.predict(x_train)

reg_train_pred = np.round(reg_train_pred).astype(int)

print('accuracy', accuracy_score(y_train, reg_train_pred))

reg_test_pred = reg.predict(x_test)
reg_test_pred = np.round(reg_test_pred).astype(int)

predicted = pd.DataFrame({
    key: test[key],
    target: reg_test_pred
})

predicted.to_csv('../input/submission.csv', index=False)
print('submission saved')
