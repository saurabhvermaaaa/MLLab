from pandas import *

from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

data = read_csv('EEG Eye State.arff', header=None)

X = data[data.columns[:-1]]
Y = data[data.columns[-1]]

Xtrain,Xtest,Ytrain,Ytest = cross_validation.train_test_split(X,Y,test_size=0.2)

classifier = GaussianNB()
classifier.fit(Xtrain,Ytrain)

print 100*classifier.score(Xtest,Ytest)
