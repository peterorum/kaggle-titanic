#!/usr/local/bin/python3

# baseline: only females survive
# kaggle score 0.7655

import sys  # pylint: disable=unused-import
import pandas as pd

# load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv("../input/test.csv")

#-------- main

test['Survived'] = test.Sex.apply(lambda x: 1 if x == 'female' else 0)

print(test.head())
print(test.describe())

predictions = test[['PassengerId', 'Survived']]

predictions.to_csv('../input/submission.csv', index=False)
