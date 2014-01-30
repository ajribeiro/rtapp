import cPickle
import mysql.connector
import dbUtils
import os
import hashlib
from sklearn.feature_extraction import FeatureHasher, DictVectorizer

def classify(title):

    with open('data/perceptron.pkl', 'rb') as fid:
        clf = cPickle.load(fid)

    mydb = dbUtils.db_access('rottentomatoes', \
            os.environ['SQLUSER'],os.environ['SQLPASS'])

    qry = "SELECT * FROM imdb_info as i inner join "
    qry += "rt_info as r on i.imdb_id = r.imdb_id "
    qry += "where i.title = \""+title+"\";"

    if ';' in title: return None

    mydb.cursor.execute(qry)
    temp_movie = [m for m in mydb.cursor]
    print temp_movie
    poster = temp_movie[0][13]
    thumb = temp_movie[0][12]
    movies = [[m[0],m[1],m[3],m[5],m[9]>=60] for m in temp_movie]

    if movies == []: return None

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


    with open('data/trainingdata.pkl', 'rb') as fid:
        training_data = cPickle.load(fid)
    
    training_data.append(movdict)

    vec = DictVectorizer()

    X_train = vec.fit_transform(training_data)

    pred = clf.predict(X_train[-1])
    print pred[0],mov[4]

    return (title, pred[0], mov[4], thumb, poster)




def regress_cast(actors,writers,director):

    with open('data/knnreg_backup.pkl', 'rb') as fid:
        clf = cPickle.load(fid)

    mydb = dbUtils.db_access('rottentomatoes', \
            os.environ['SQLUSER'],os.environ['SQLPASS'])

    print director

    qry = "SELECT ratio,total FROM director_stats WHERE "
    qry += "director = %s;" % ("\""+director+"\"")
    print qry
    director_stats = mydb.query(qry)
    directorsum = 0.
    total = 1e-6
    for w in director_stats:
        directorsum += w[0]*w[1]
        total += w[1]
    directorsum /= float(total)


    actor_ids = ["\""+a+"\"" for a in actors]
    actorlist = "("
    for w in actor_ids:
        actorlist += w+","
    actorlist = actorlist[:-1]+")"
    qry = "SELECT ratio,total FROM actor_stats WHERE "
    qry += "actor_name IN %s;" % actorlist
    print qry 
    actor_stats = mydb.query(qry)
    actorsum = 0.
    total = 1e-6
    for w in actor_stats:
        actorsum += w[0]*w[1]
        total += w[1]
    actorsum /= float(total)



    writer_ids = ("\""+a+"\"" for a in writers)
    writerlist = "("
    for w in writer_ids:
        writerlist += w+","
    writerlist = writerlist[:-1]+")"
    qry = "SELECT ratio,total FROM writer_stats WHERE "
    qry += "writer_name IN %s;" % writerlist
    print qry 
    writer_stats = mydb.query(qry)
    writersum = 0.
    total = 1e-6
    for w in writer_stats:
        writersum += w[0]*w[1]
        total += w[1]
    writersum /= float(total)

    prediction = clf.predict([actorsum,directorsum,writersum])[0]


    return prediction


def regress_title(title):

    with open('data/knnreg_backup.pkl', 'rb') as fid:
        clf = cPickle.load(fid)

    mydb = dbUtils.db_access('rottentomatoes', \
            os.environ['SQLUSER'],os.environ['SQLPASS'])

    qry = "SELECT * FROM imdb_info as i inner join "
    qry += "rt_info as r on i.imdb_id = r.imdb_id "
    qry += "where i.title = %s;" % ("\""+title+"\"")

    if ';' in title: return None

    movies = mydb.query(qry)[0]

    # movies = [[m[0],m[1],m[3],m[5],m[9]>=60] for m in temp_movie]

    if movies == []: return None


    poster = movies[13]
    thumb = movies[12]

    imdb_id = movies[0]
    director = movies[3]

    print director

    qry = "SELECT ratio,total FROM director_stats WHERE "
    qry += "director = %s;" % ("\""+director+"\"")
    print qry 
    director_stats = mydb.query(qry)
    directorsum = 0.
    total = 1e-6
    for w in director_stats:
        directorsum += w[0]*w[1]
        total += w[1]
    directorsum /= float(total)



    qry = "SELECT actor_id,nam FROM casts WHERE "
    qry += "imdb_id = %s;" % ("\""+imdb_id+"\"")
    actors = mydb.query(qry)
    actor_ids = ["\""+a[0]+"\"" for a in actors]
    actornames = [a[1] for a in actors]

    actorlist = "("
    for w in actor_ids:
        actorlist += w+","
    actorlist = actorlist[:-1]+")"
    qry = "SELECT ratio,total FROM actor_stats WHERE "
    qry += "actor_id IN %s;" % actorlist
    print qry 
    actor_stats = mydb.query(qry)
    actorsum = 0.
    total = 1e-6
    for w in actor_stats:
        actorsum += w[0]*w[1]
        total += w[1]
    actorsum /= float(total)



    qry = "SELECT writer_id,writer_name FROM writers WHERE "
    qry += "imdb_id = %s;" % ("\""+imdb_id+"\"")
    writers = mydb.query(qry)
    writer_ids = ("\""+a[0]+"\"" for a in writers)
    writernames = [a[1] for a in writers]

    writerlist = "("
    for w in writer_ids:
        writerlist += w+","
    writerlist = writerlist[:-1]+")"
    qry = "SELECT ratio,total FROM writer_stats WHERE "
    qry += "writer_id IN %s;" % writerlist
    print qry 
    writer_stats = mydb.query(qry)
    writersum = 0.
    total = 1e-6
    for w in writer_stats:
        writersum += w[0]*w[1]
        total += w[1]
    writersum /= float(total)

    prediction = clf.predict([actorsum,directorsum,writersum])[0]
    qry = "SELECT critic FROM rt_info WHERE "
    qry += "imdb_id = %s;" % ("\""+imdb_id+"\"")
    actual = mydb.query(qry)[0]

    return {'prediction':prediction, 'actual':actual, \
        'cast':actornames, 'director':director, 'writers':writernames,
        'thumb':thumb, 'poster':poster}
