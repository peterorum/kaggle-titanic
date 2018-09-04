# baseline: all died
# kaggle score 0.626

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

result = 0

train['predicted'] = result

score = accuracy_score(train[target], train.predicted)
print('score', score)

test['predicted'] = result

predicted = pd.DataFrame({
    "PassengerId": test.PassengerId,
    target: test.predicted
})

predicted.to_csv('submission.csv', index=False)

print('%.0f mins' % ((time() - start_time) / 60))
