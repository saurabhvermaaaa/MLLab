import pandas as pd

from sklearn.svm import SVC
from sklearn import cross_validation

data = pd.read_table('Skin_NonSkin.txt', names = ['R', 'G', 'B', 'Class'])
classifier = SVC(verbose=True)

X = data[['R', 'G', 'B']]
Y = data['Class']

Xtrain,Xtest,Ytrain,Ytest = cross_validation.train_test_split(X,Y,test_size=0.96,random_state=0)

classifier.fit(Xtrain,Ytrain)

print classifier.score(Xtest[:50000],Ytest[:50000])