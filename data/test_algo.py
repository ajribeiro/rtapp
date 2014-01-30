import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import pylab as pl
import mysql.connector
from mysql.connector import errorcode
import dbUtils
import os
import hashlib
import random
import cPickle
from sklearn.feature_selection import RFE

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import FeatureHasher, DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics


class dataset(object):
    def __init__(self):
        self.data = []
        self.target = []
        self.raw = []
        self.ids = []

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


mydb = dbUtils.db_access('rottentomatoes', \
        os.environ['SQLUSER'],os.environ['SQLPASS'])


qry = "SELECT * FROM imdb_info as i inner join "
qry += "rt_info as r on i.imdb_id = r.imdb_id;"

mydb.cursor.execute(qry)
movies = [[m[0],m[1],m[3],m[5],m[9]>=60] for m in mydb.cursor]

for mov in movies:
    [id,title,director,budget,rat] = mov
    qry = "SELECT actor_id,nam from casts where imdb_id = "+"\""+id+"\";"
    mydb.cursor.execute(qry)
    actors = [m for m in mydb.cursor]
    largeactors=[]
    for a in actors:
        qry = "SELECT COUNT(*) from casts where actor_id = "+"\""+a[0]+"\";"
        mydb.cursor.execute(qry)
        cnt = [m for m in mydb.cursor]
        if cnt[0][0] > 0:
            largeactors.append(a)
    mov.append(largeactors)

n = len(movies)

print 'Number',n

n_trn = int(n*1.)
n_tst = n - n_trn

random.shuffle(movies)

training_data = dataset()
testing_data = dataset()

dirhash = {}
for m in movies:
    hash_object = hashlib.sha224(m[2])
    hex_dig = hash_object.hexdigest()
    dirhash[m[2]] = hex_dig

for i in range(n_trn):
    training_data.ids.append(movies[i][0])
    training_data.raw.append(movies[i])
    if movies[i][4]: training_data.target.append(1)
    else: training_data.target.append(0)
    training_data.data.append({dirhash[movies[i][2]]:1})
    for a in movies[i][5]:
        training_data.data[i][a[0]] = 1
    # training_data.data[i]['budget'] = movies[i][3]
    
for i in range(n_trn,n):
    testing_data.ids.append(movies[i][0])
    testing_data.raw.append(movies[i])
    if movies[i][4]: testing_data.target.append(1)
    else: testing_data.target.append(0)
    testing_data.data.append({dirhash[movies[i][2]]:1})
    for a in movies[i][5]:
        testing_data.data[i-n_trn][a[0]] = 1
    # testing_data.data[i-n_trn]['budget'] = movies[i][3]


# print 'start features;'
# features = []
# for d in training_data.data:
#     for f in d:
#         if type(f) != int and f not in features:
#             features.append(f)
# print 'done features;'

# hasher = FeatureHasher(input_type='string',non_negative=True)
# print 'start featurizing'
# training = []
# for d in training_data.data:
#     data_dict = {}
#     for f in features:
#         if f in d:
#             data_dict[f] = 1
#         else:
#             data_dict[f] = 0
#     data_dict['budget'] = training_data.data[-1]
#     training.append(data_dict)
# print 'done featurizing'

# X_train = hasher.transform(training_data.data)
# y_train = training_data.target

# X_test = hasher.transform(testing_data.data)
# y_test = testing_data.target

testfile = open('trainingset.txt','w')
vec = DictVectorizer()
for d in training_data.ids[:int(n*.7)]:
    testfile.write("\""+d+"\"\n")
testfile.close()

X_train = vec.fit_transform(training_data.data)
with open('trainingdata.pkl', 'wb') as fid:
    cPickle.dump(training_data.data, fid)
y_train = training_data.target

train_size = .8
X_test = X_train[int(n*train_size):]
y_test = y_train[int(n*train_size):]

X_train = X_train[:int(n*train_size)]
y_train = y_train[:int(n*train_size)]
# X_test = vec.fit_transform(testing_data.data)
# y_test = testing_data.target


###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    # rfe = RFE(estimator=clf, n_features_to_select=1000, step=1)
    # rfe.fit(X_train, y_train)
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.f1_score(y_test, pred)
    print("f1-score:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        # if opts.print_top10 and feature_names is not None:
        #     print("top 10 keywords per class:")
        #     for i, category in enumerate(categories):
        #         top10 = np.argsort(clf.coef_[i])[-10:]
        #         print(trim("%s: %s"
        #               % (category, " ".join(feature_names[top10]))))
        print()

    # if opts.print_report:
    #     print("classification report:")
    #     print(metrics.classification_report(y_test, pred,
    #                                         target_names=categories))

    # if opts.print_cm:
    #     print("confusion matrix:")
    #     print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time
    

results = []
# for clf, name in (
#         (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
#         (Perceptron(n_iter=50), "Perceptron"),
#         (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
#         (KNeighborsClassifier(n_neighbors=10), "kNN")):
#     print('=' * 80)
#     print(name)
#     results.append(benchmark(clf))

# for penalty in ["l2", "l1"]:
#     print('=' * 80)
#     print("%s penalty" % penalty.upper())
#     # Train Liblinear model
#     results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
#                                             dual=False, tol=1e-3)))

#     # Train SGD model
#     results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
#                                            penalty=penalty)))

# # Train SGD with Elastic Net penalty
# print('=' * 80)
# print("Elastic-Net penalty")
# results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
#                                        penalty="elasticnet")))

# # Train NearestCentroid without threshold
# print('=' * 80)
# print("NearestCentroid (aka Rocchio classifier)")
# results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
# results.append(benchmark(MultinomialNB(alpha=.01)))
clf = BernoulliNB(alpha=.01,fit_prior=True)
results.append(benchmark(clf))
with open('perceptron.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)


class L1LinearSVC(LinearSVC):

    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1",
                                      dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)

print('=' * 80)
print("LinearSVC with L1-based feature selection")
results.append(benchmark(L1LinearSVC()))


# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

pl.figure(figsize=(12,8))
pl.title("Score")
pl.barh(indices, score, .2, label="score", color='r')
pl.barh(indices + .3, training_time, .2, label="training time", color='g')
pl.barh(indices + .6, test_time, .2, label="test time", color='b')
pl.yticks(())
pl.legend(loc='best')
pl.subplots_adjust(left=.25)
pl.subplots_adjust(top=.95)
pl.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    pl.text(-.3, i, c)

pl.show()