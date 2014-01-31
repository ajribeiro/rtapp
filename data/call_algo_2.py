import cPickle
import mysql.connector
from mysql.connector import errorcode
import dbUtils
import os
import hashlib
from sklearn.feature_extraction import FeatureHasher, DictVectorizer

with open('featurelist.pkl', 'rb') as fid:
    featurelist = cPickle.load(fid)

with open('targets.pkl', 'rb') as fid:
    targets = cPickle.load(fid)

with open('movielist.pkl', 'rb') as fid:
    mlist = cPickle.load(fid)

with open('knnreg_backup.pkl', 'rb') as fid:
    clf = cPickle.load(fid)

training_files = {}
f = open('trainingset.txt','r')
for line in f:
    training_files[line[1:-2]] = 1
f.close()


vec = DictVectorizer()
trainsuc,trainfail,testsuc,testfail = 0,0,0,0
for i in range(len(featurelist)):

    X_train = vec.fit_transform([featurelist[i]])

    pred = clf.predict(X_train[-1])


    print mlist[i],pred[0],targets[i]
    if mlist[i] in training_files:
        if (pred[0]==1 and targets[i]) or (pred[0]==0 and not targets[i]):
            print 'train correct!'
            trainsuc += 1
        else:
            print 'train fail'
            trainfail += 1
        print 'train',trainsuc/float(trainsuc+trainfail)
    else:
        if (pred[0]==1 and targets[i]) or (pred[0]==0 and not targets[i]):
            print 'test correct!'
            testsuc += 1
        else:
            print 'test fail'
            testfail += 1
        print 'test',testsuc/float(testsuc+testfail)

