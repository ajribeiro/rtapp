# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import pylab as pl
import os
import pandas as pd
import cPickle
import random

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import DictVectorizer
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
import dbUtils


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                                        format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
                            action="store_true", dest="print_report",
                            help="Print a detailed classification report.")
op.add_option("--chi2_select",
                            action="store", type="int", dest="select_chi2",
                            help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
                            action="store_true", dest="print_cm",
                            help="Print the confusion matrix.")
op.add_option("--top10",
                            action="store_true", dest="print_top10",
                            help="Print ten most discriminative terms per class"
                                     " for every classifier.")
op.add_option("--all_categories",
                            action="store_true", dest="all_categories",
                            help="Whether to use all categories or not.")
op.add_option("--use_hashing",
                            action="store_true",
                            help="Use a hashing vectorizer.")
op.add_option("--n_features",
                            action="store", type=int, default=2 ** 16,
                            help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
                            action="store_true",
                            help="Remove newsgroup information that is easily overfit: "
                                     "headers, signatures, and quoting.")

(opts, args) = op.parse_args()
if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)

print(__doc__)
op.print_help()
print()


mydb = dbUtils.db_access('rottentomatoes', \
        os.environ['SQLUSER'],os.environ['SQLPASS'])

featurelist = []
targets = []
regval = []

movielist = mydb.query("SELECT * FROM rt_info;")
movielist = pd.DataFrame(movielist, \
    columns=['imdb_id','rt_id','title','critic','aud','cast', 'thumb', 'orig'])

imdb_info = mydb.query("SELECT imdb_id,director,budget FROM imdb_info;")
imdb_info_df = pd.DataFrame(imdb_info, \
    columns=['imdb_id','director','budget'])


qry = "SELECT actor_id,imdb_id FROM casts;"
actorlist = mydb.query(qry)
casts_df = pd.DataFrame(actorlist, \
    columns=['actor_id','imdb_id'])

qry = "SELECT actor_id,ratio,total FROM actor_stats;"
actor_stats = mydb.query(qry)
actor_stats_df = pd.DataFrame(actor_stats, \
    columns=['actor_id','ratio','total'])


qry = "SELECT director,ratio,total FROM director_stats;"
director_stats = mydb.query(qry)
director_stats_df = pd.DataFrame(director_stats, \
    columns=['director','ratio','total'])


qry = "SELECT imdb_id,writer_id FROM writers;"
writerlist = mydb.query(qry)
writer_df = pd.DataFrame(writerlist, \
    columns=['imdb_id','writer_id'])

qry = "SELECT writer_id,ratio,total FROM writer_stats;"
writer_stats = mydb.query(qry)
writer_stats_df = pd.DataFrame(writer_stats, \
    columns=['writer_id','ratio','total'])

ids = []
mlist = movielist.imdb_id.tolist()
random.shuffle(mlist)

allmovies = []
for m in mlist:
    imdb_id = m

    rating = movielist[movielist.imdb_id == imdb_id].critic.values[0]
    if rating < 0: continue

    allmovies.append(m)
    ids.append(imdb_id)
    print(imdb_id)
    # qry = "SELECT actor_id FROM casts where "
    # qry += "imdb_id = %s;" % ("\""+imdb_id+"\"")
    # actorlist = mydb.query(qry)
    actorlist = casts_df[casts_df.imdb_id == imdb_id].actor_id.tolist()
    actorsum = 0.
    totals = 1e-6
    for actor_id in actorlist:
        actor = actor_stats_df[actor_stats_df.actor_id == int(actor_id)]
        ratio = actor.ratio.values[0]
        total = actor.total.values[0]
        totals += total
        actorsum += ratio*total
    actorsum /= totals

    # qry = "SELECT writer_id FROM writers where "
    # qry += "imdb_id = %s;" % ("\""+imdb_id+"\"")
    # writerlist = mydb.query(qry)
    writersum = 0.
    totals = 1e-6
    writerlist = writer_df[writer_df.imdb_id == imdb_id].writer_id.tolist()
    for writer_id in writerlist:
        # writer_id = w[0]
        # qry = "SELECT ratio,total FROM writer_stats "
        # qry +="where writer_id = %s;" % ("\""+writer_id+"\"")
        # writer_stats = mydb.query(qry)
        # writersum += writer_stats[0][0]*writer_stats[0][1]
        writer = writer_stats_df[writer_stats_df.writer_id == writer_id]
        ratio = writer.ratio.values[0]
        total = writer.total.values[0]
        totals += total
        writersum += ratio*total
    writersum /= totals

    # qry = "SELECT director FROM imdb_info where "
    # qry += "imdb_id = %s;" % ("\""+imdb_id+"\"")
    # directorlist = mydb.query(qry)
    directorlist = imdb_info_df[imdb_info_df.imdb_id == imdb_id].director
    directorsum = 0.
    totals = 1e-6
    for director_id in directorlist:
        direc = director_stats_df[director_stats_df.director == director_id]
        ratio = direc.ratio.values[0]
        total = direc.total.values[0]
        totals += total
        directorsum += ratio*total
    directorsum /= totals

    # qry = "SELECT critic from rt_info where "
    # qry += "imdb_id = %s;" % ("\""+imdb_id+"\"")
    # rating = mydb.query(qry)
    f = {'actorsum':actorsum, 'writersum':writersum, \
        'directorsum':directorsum}
    featurelist.append(f)
    regval.append(rating)
    if rating >= 60:
        targets.append(1)
    else:
        targets.append(0)

print('done dataing')


# ###############################################################################
# # Load some categories from the training set
# if opts.all_categories:
#     categories = None
# else:
#     categories = [
#         'alt.atheism',
#         'talk.religion.misc',
#         'comp.graphics',
#         'sci.space',
#     ]

# if opts.filtered:
#     remove = ('headers', 'footers', 'quotes')
# else:
#     remove = ()

# print("Loading 20 newsgroups dataset for categories:")
# print(categories if categories else "all")

# data_train = fetch_20newsgroups(subset='train', categories=categories,
#                                 shuffle=True, random_state=42,
#                                 remove=remove)

# data_test = fetch_20newsgroups(subset='test', categories=categories,
#                                shuffle=True, random_state=42,
#                                remove=remove)
# print('data loaded')

# categories = data_train.target_names    # for case categories == None


# def size_mb(docs):
#     return sum(len(s.encode('utf-8')) for s in docs) / 1e6

# data_train_size_mb = size_mb(data_train.data)
# data_test_size_mb = size_mb(data_test.data)

# print("%d documents - %0.3fMB (training set)" % (
#     len(data_train.data), data_train_size_mb))
# print("%d documents - %0.3fMB (test set)" % (
#     len(data_test.data), data_test_size_mb))
# print("%d categories" % len(categories))
# print()

# # split a training set and a test set
# y_train, y_test = data_train.target, data_test.target

# print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
# if opts.use_hashing:
#     vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
#                                    n_features=opts.n_features)
#     X_train = vectorizer.transform(data_train.data)
# else:
#     vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
#                                  stop_words='english')
#     X_train = vectorizer.fit_transform(data_train.data)


with open('movielist.pkl', 'wb') as fid:
    cPickle.dump(allmovies, fid)

with open('featurelist.pkl', 'wb') as fid:
    cPickle.dump(featurelist, fid)

with open('targets.pkl', 'wb') as fid:
    cPickle.dump(targets, fid)

with open('regval.pkl', 'wb') as fid:
    cPickle.dump(regval, fid)


# with open('featurelist.pkl', 'rb') as fid:
#     featurelist = cPickle.load(fid)


# with open('targets.pkl', 'rb') as fid:
#     targets = cPickle.load(fid)


n_trn = int(.8*len(featurelist))
n_tst = len(featurelist)-n_trn
vec = DictVectorizer()
X_train = vec.fit_transform(featurelist[:n_trn])
X_test = vec.fit_transform(featurelist[n_trn:])
y_train = targets[:n_trn]
y_test = targets[n_trn:]


# testfile = open('trainingset.txt','w')
# for d in ids[:n_trn]:
#     testfile.write("\""+d+"\"\n")
# testfile.close()



# duration = time() - t0
# print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
# print("n_samples: %d, n_features: %d" % X_train.shape)
# print()

# print("Extracting features from the test dataset using the same vectorizer")
# t0 = time()
# # X_test = vectorizer.transform(data_test.data)
# duration = time() - t0
# print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
# print("n_samples: %d, n_features: %d" % X_test.shape)
# print()

if opts.select_chi2:
        print("Extracting %d best features by a chi-squared test" %
                    opts.select_chi2)
        t0 = time()
        ch2 = SelectKBest(chi2, k=opts.select_chi2)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        print("done in %fs" % (time() - t0))
        print()


def trim(s):
        """Trim string to fit on terminal (assuming 80-column display)"""
        return s if len(s) <= 80 else s[:77] + "..."


# # mapping from integer feature name to original token string
# if opts.use_hashing:
#         feature_names = None
# else:
#         feature_names = np.asarray(vectorizer.get_feature_names())


###############################################################################
# Benchmark classifiers
def benchmark(clf):
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
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

                if opts.print_top10 and feature_names is not None:
                        print("top 10 keywords per class:")
                        for i, category in enumerate(categories):
                                top10 = np.argsort(clf.coef_[i])[-10:]
                                print(trim("%s: %s"
                                            % (category, " ".join(feature_names[top10]))))
                print()

        if opts.print_report:
                print("classification report:")
                print(metrics.classification_report(y_test, pred,
                                                                                        target_names=categories))

        if opts.print_cm:
                print("confusion matrix:")
                print(metrics.confusion_matrix(y_test, pred))

        print()
        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, train_time, test_time


results = []
for k in range(5,6):
    clf = KNeighborsClassifier(n_neighbors=k)
    name = 'kNN %d' % k
    # for (clf, name) in [(KNeighborsClassifier(n_neighbors=10), "kNN")]:
    #         print('=' * 80)
    #         print(name)
    results.append(benchmark(clf))

# with open('knnclass.pkl', 'wb') as fid:
#     cPickle.dump(clf, fid)


# for penalty in ["l2", "l1"]:
#         print('=' * 80)
#         print("%s penalty" % penalty.upper())
#         # Train Liblinear model
#         results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
#                                                                                         dual=False, tol=1e-3)))

#         # Train SGD model
#         results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
#                                                                                      penalty=penalty)))

# # Train SGD with Elastic Net penalty
# print('=' * 80)
# print("Elastic-Net penalty")
# results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
#                                                                              penalty="elasticnet")))

# # Train NearestCentroid without threshold
# print('=' * 80)
# print("NearestCentroid (aka Rocchio classifier)")
# results.append(benchmark(NearestCentroid()))

# # Train sparse Naive Bayes classifiers
# print('=' * 80)
# print("Naive Bayes")
# results.append(benchmark(MultinomialNB(alpha=.01)))
# results.append(benchmark(BernoulliNB(alpha=.01)))


# class L1LinearSVC(LinearSVC):

#         def fit(self, X, y):
#                 # The smaller C, the stronger the regularization.
#                 # The more regularization, the more sparsity.
#                 self.transformer_ = LinearSVC(penalty="l1",
#                                                                             dual=False, tol=1e-3)
#                 X = self.transformer_.fit_transform(X, y)
#                 return LinearSVC.fit(self, X, y)

#         def predict(self, X):
#                 X = self.transformer_.transform(X)
#                 return LinearSVC.predict(self, X)

# print('=' * 80)
# print("LinearSVC with L1-based feature selection")
# results.append(benchmark(L1LinearSVC()))


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