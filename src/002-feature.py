# females survived 0.787
# kaggle score 0.7655

import sys  # pylint: disable=unused-import
from time import time
import os
import numpy as np  # pylint: disable=unused-import
import pandas as pd
from sklearn.metrics import accuracy_score

is_kaggle = os.environ['HOME'] == '/tmp'

zipext = ''  # if is_kaggle else '.zip'

# load data
train = pd.read_csv(f'../input/train.csv{zipext}')
test = pd.read_csv(f'../input/test.csv{zipext}')

#-------- main

start_time = time()

target = 'Survived'

train['predicted'] = train.Sex.apply(lambda x: 1 if x == 'female' else 0)

score = accuracy_score(train[target], train.predicted)
print('score to maximize', score)

test['predicted'] = test.Sex.apply(lambda x: 1 if x == 'female' else 0)


predicted = pd.DataFrame({
    "PassengerId": test.PassengerId,
    target: test.predicted
})

predicted.to_csv('submission.csv', index=False)

print('%.0f mins' % ((time() - start_time) / 60))
