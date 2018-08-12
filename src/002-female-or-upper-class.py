#!/usr/local/bin/python3

# only females or upper class survive
# kaggle score 0.70813

import sys  # pylint: disable=unused-import
import pandas as pd

# load data
train = pd.read_csv('../data/train.csv')
test = pd.read_csv("../data/test.csv")

#-------- main

test['Survived'] = test.apply(lambda x: 1 if (x.Sex == 'female') | (x.Pclass == 1) else 0, axis=1)

print(test.head())
print(test.describe())

predictions = test[['PassengerId', 'Survived']]

predictions.to_csv('../data/submission.csv', index=False)
