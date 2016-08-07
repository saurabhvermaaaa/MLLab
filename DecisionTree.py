import numpy as np
import pandas as pd

column_names = ["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight", "shell weight", "rings"]

data = pd.read_csv("abalone.data.txt", names=column_names)

out = data.rings.values
del data["rings"]
for val in ['M', 'F', 'I']:
    data[val] = data["sex"] == val
del data["sex"]

inp = data.values.astype(np.float)

from sklearn import cross_validation
train_in, test_in, train_out, test_out = cross_validation.              train_test_split(inp, out)

from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=6)
model = model.fit(train_in, train_out)
print("%s" % model.score(test_in, test_out))
tree.export_graphviz(model, out_file='tree.dot')
