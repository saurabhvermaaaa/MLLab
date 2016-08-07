from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
import pandas as pd

column_names = ['class', 'left-weight', 'left-distance', 'right-weight', 'right-distance']

data = pd.read_csv('balance-scale.data', names=column_names)

out = data['class']
del data['class']
inp = data.values

train_in, test_in, train_out, test_out = cross_validation.train_test_split(inp, out)
model = KNeighborsClassifier(n_neighbors=10)
model.fit(train_in, train_out)

print ("%s" % model.score(test_in, test_out))
