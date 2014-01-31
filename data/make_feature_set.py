import dbUtils
import os
import pandas as pd

mydb = dbUtils.db_access('rottentomatoes', \
        os.environ['SQLUSER'],os.environ['SQLPASS'])


results = mydb.query("SELECT * FROM rt_info where critic > -1;")
df = pd.DataFrame(results, \
    columns=['imdb_id','rt_id','title','critic','aud','cast', 'thumb', 'orig'])


all_actors = mydb.query("SELECT imdb_id,actor_id,nam FROM casts")

actordict = {}
for [imdb_id,actor_id,actor_name] in all_actors:
    if actor_id not in actordict:
        actordict[actor_id] = {'name': actor_name, 'fresh': 0, 'rotten': 0}

    rating = df[df['imdb_id'] == imdb_id]['critic']

    if rating >= 60:
        actordict[actor_id]['fresh'] += 1
    else:
        actordict[actor_id]['rotten'] += 1

for key,val in actordict.iteritems():
    qry = "INSERT INTO actor_stats "
    qry += "(actor_id, actor_name, ratio, total) "
    qry += "VALUES (%s,  %s,  %f, %d);" % \
        ("\""+key+"\"", "\""+val['name']+"\"", \
        float(val['fresh'])/(val['fresh']+val['rotten']),
        val['fresh']+val['rotten'])

    print qry;

    mydb.cursor.execute(qry)
    mydb.cnx.commit()



# actordict = {}
# for (imdb_id,actor_id,name,billed) in all_actors:
#     if actor_id not in actordict:
#         actordict[actor_id] = {'name':name, 'fresh':0, 'rotten': 0}

#     print "SELECT critic FROM rt_info WHERE imdb_id = %s;" % ("\""+imdb_id+"\"")
#     rating = mydb.query("SELECT critic FROM rt_info WHERE imdb_id = %s;" % ("\""+imdb_id+"\""))
#     if not rating: continue
#     rating = rating[0][0]
#     if rating >= 60:
#         actordict[actor_id]['fresh'] += 1
#     else:
#         actordict[actor_id]['rotten'] += 1

# for key,val in actordict.iteritems():
#     if val['fresh']+val['rotten'] <= 0: continue
#     qry = "INSERT INTO actor_stats2 "
#     qry += "(actor_id, actor_name, ratio, total) "
#     qry += "VALUES (%s,  %s,  %f,  %d);" % \
#         ("\""+key+"\"", "\""+val['name']+"\"", \
#         float(val['fresh'])/(val['fresh']+val['rotten']), \
#         val['fresh']+val['rotten'])

#     print qry;

#     mydb.cursor.execute(qry)
#     mydb.cnx.commit()


dirdict = {}
all_dirs = mydb.query("SELECT imdb_id,director from imdb_info;");
for [imdb_id,director] in all_dirs:
    if director not in dirdict:
        dirdict[director] = {'fresh': 0, 'rotten': 0}
    rating = df[df['imdb_id'] == imdb_id]['critic']

    if rating >= 60:
        dirdict[director]['fresh'] += 1
    else:
        dirdict[director]['rotten'] += 1

for key,val in dirdict.iteritems():
    qry = "INSERT INTO director_stats "
    qry += "(director, ratio, total) "
    qry += "VALUES (%s, %f, %d);" % \
        ("\""+key+"\"", \
        float(val['fresh'])/(val['fresh']+val['rotten']), \
        val['fresh']+val['rotten'])

    print qry;

    mydb.cursor.execute(qry)
    mydb.cnx.commit()

writerdict = {}
all_dirs = mydb.query("SELECT * from writers;");
for [imdb_id,writer_id,writer_name] in all_dirs:
    if writer_id not in writerdict:
        writerdict[writer_id] = {'name': writer_name, 'fresh': 0, 'rotten': 0}

    rating = df[df['imdb_id'] == imdb_id]['critic']

    if rating >= 60:
        writerdict[writer_id]['fresh'] += 1
    else:
        writerdict[writer_id]['rotten'] += 1

for key,val in writerdict.iteritems():
    qry = "INSERT INTO writer_stats "
    qry += "(writer_id, writer_name, ratio, total) "
    qry += "VALUES (%s,  %s,  %f, %d);" % \
        ("\""+key+"\"", "\""+val['name']+"\"", \
        float(val['fresh'])/(val['fresh']+val['rotten']),
        val['fresh']+val['rotten'])

    print qry;

    mydb.cursor.execute(qry)
    mydb.cnx.commit()