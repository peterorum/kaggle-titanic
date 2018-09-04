# use xgboost
# accuracy 0.874
# kaggle score 0.7703

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
train = pd.read_csv('../data/train.csv')
test = pd.read_csv("../data/test.csv")


# fill missing values with mean
mean_age = train.Age.mean()
train.Age.fillna(mean_age, inplace=True)
test.Age.fillna(mean_age, inplace=True)

train.Fare.fillna(train.Fare.mean(), inplace=True)
test.Fare.fillna(test.Fare.mean(), inplace=True)

# need to combine ot encode

all_data = pd.concat([train, test])

onehot_encoder = ce.OneHotEncoder(cols=['Sex', 'Embarked'])
all_data = onehot_encoder.fit_transform(all_data)

# separate again
train = all_data[all_data.PassengerId.isin(train.PassengerId)]
test = all_data[all_data.PassengerId.isin(test.PassengerId)]

#-------- main

x_train = train.drop(['Survived', 'PassengerId', 'SibSp', 'Parch', 'Name', 'Ticket', 'Cabin'], axis=1)
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

predicted.to_csv('../data/submission.csv', index=False)
print('submission saved')
