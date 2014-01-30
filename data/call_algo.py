import cPickle
import mysql.connector
from mysql.connector import errorcode
import dbUtils
import os
import hashlib
from sklearn.feature_extraction import FeatureHasher, DictVectorizer

training_files = {}
f = open('trainingset.txt','r')
for line in f:
    training_files[line[1:-2]] = 1
f.close()

with open('perceptron.pkl', 'rb') as fid:
    clf = cPickle.load(fid)

mydb = dbUtils.db_access('rottentomatoes', \
        os.environ['SQLUSER'],os.environ['SQLPASS'])

qry = "SELECT title FROM imdb_info"

mydb.cursor.execute(qry)
alltitles = [m[0] for m in mydb.cursor]

trainsuc,trainfail = 0,0
testsuc,testfail=0,0
for title in alltitles:

    qry = "SELECT * FROM imdb_info as i inner join "
    qry += "rt_info as r on i.imdb_id = r.imdb_id "
    qry += "where i.title = \""+title+"\";"

    mydb.cursor.execute(qry)
    movies = [[m[0],m[1],m[3],m[5],m[9]>=60] for m in mydb.cursor]

    if movies == []: continue

    mov = movies[0]

    [id,title,director,budget,rat] = mov
    qry = "SELECT actor_id,nam from casts where imdb_id = "+"\""+id+"\";"
    mydb.cursor.execute(qry)
    actors = [m for m in mydb.cursor]
    mov.append(actors)

    movdict = {}
    for a in mov[5]:
        movdict[a[0]] = 1

    # movdict['budget'] = mov[3]
    hash_object = hashlib.sha224(mov[2])
    hex_dig = hash_object.hexdigest()
    movdict[hex_dig] = 1


    with open('trainingdata.pkl', 'rb') as fid:
        training_data = cPickle.load(fid)
    
    training_data.append(movdict)

    vec = DictVectorizer()

    X_train = vec.fit_transform(training_data)

    pred = clf.predict(X_train[-1])

    print title,pred[0],mov[4]
    if mov[0] in training_files:
        if (pred[0]==1 and mov[4]) or (pred[0]==0 and not mov[4]):
            print 'train correct!'
            trainsuc += 1
        else:
            print 'train fail'
            trainfail += 1
        print 'train',trainsuc/float(trainsuc+trainfail)
    else:
        if (pred[0]==1 and mov[4]) or (pred[0]==0 and not mov[4]):
            print 'test correct!'
            testsuc += 1
        else:
            print 'test fail'
            testfail += 1
        print 'test',testsuc/float(testsuc+testfail)

