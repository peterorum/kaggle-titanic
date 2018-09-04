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
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath("../../utils"))
from pjo_plot import plot_histograms  # pylint: disable=import-error,wrong-import-position,unused-import

warnings.simplefilter(action='ignore')
pd.options.display.float_format = '{:.4f}'.format

# load data
train = pd.read_csv('../data/train.csv')
test = pd.read_csv("../data/test.csv")

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

# feature selection via variance
train_numeric = train[numeric_cols].fillna(0)
select_features = VarianceThreshold(threshold=0.2)
select_features.fit(train_numeric)
numeric_cols = train_numeric.columns[select_features.get_support(indices=True)].tolist()

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

# replace missing categoricals with mode
# for col in categorical_cols:
#     if all_data[col].isnull().sum() > 0:
#         mode = all_data[col].mode()[0]
#         all_data[col].fillna(mode, inplace=True)

# replace missing values with mean
for col in numeric_cols:
    if all_data[col].isna().any():
        all_data[col].fillna(all_data[col].mean(), inplace=True)

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

predicted.to_csv('../data/submission.csv', index=False)
print('submission saved')
