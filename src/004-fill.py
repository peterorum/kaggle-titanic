# fillna 0.8675
# kaggle score 0.784 (worse)
# maximize score

import sys  # pylint: disable=unused-import
from time import time
import os
import numpy as np  # pylint: disable=unused-import
import pandas as pd
from sklearn.metrics import accuracy_score
import xgboost as xgb

is_kaggle = os.environ['HOME'] == '/tmp'

zipext = ''  # if is_kaggle else '.zip'

pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_columns', None)

# load data
train = pd.read_csv(f'../input/train.csv{zipext}')
test = pd.read_csv(f'../input/test.csv{zipext}')

#-------- main

start_time = time()

key = 'PassengerId'
target = 'Survived'

numeric_cols = [col for col in train.columns
                if (train[col].dtype == 'int64') | (train[col].dtype == 'float64')]

categorical_cols = [col for col in train.columns if train[col].dtype == 'object']

# replace missing numericals with mean
for col in numeric_cols:
    if train[col].isna().any():
        mean = train[col].mean()

        train[col].fillna(mean, inplace=True)

        if col != target:
            test[col].fillna(mean, inplace=True)

# replace missing categoricals with mode
for col in categorical_cols:
    if train[col].isna().any():
        mode = train[col].mode()[0]

        train[col].fillna(mode, inplace=True)

        if col != target:
            test[col].fillna(mode, inplace=True)

# encode categoricals so all numeric

# drop if too many values
max_categories = 20
many_value_cols = [col for col in categorical_cols if train[col].nunique() >= max_categories]
few_value_cols = [col for col in categorical_cols if train[col].nunique() < max_categories]

train = train.drop(many_value_cols, axis=1)

# encode
train = pd.get_dummies(train, columns=few_value_cols)
test = pd.get_dummies(test, columns=few_value_cols)

# model

x_train = train.drop([key, target], axis=1)
y_train = train[target]
x_test = test[x_train.columns]

reg = xgb.XGBRegressor()
reg.fit(x_train, y_train)

train_pred = reg.predict(x_train)
# 0 or 1
train_pred = np.round(train_pred).astype(int)

print('accuracy', accuracy_score(y_train, train_pred))

test_pred = reg.predict(x_test)
test_pred = np.round(test_pred).astype(int)

predicted = pd.DataFrame({
    "PassengerId": test[key],
    "Survived": test_pred
})

predicted.to_csv('submission.csv', index=False)

print('%.0f mins\a' % ((time() - start_time) / 60))
