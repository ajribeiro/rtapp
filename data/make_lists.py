import dbUtils
import json
import os

mydb = dbUtils.db_access('rottentomatoes', \
        os.environ['SQLUSER'],os.environ['SQLPASS'])

qry = "select * from imdb_info inner join rt_info "
qry += "on imdb_info.imdb_id = rt_info.imdb_id "
qry += "order by budget desc;"

movies = mydb.query(qry)

titles,ids,posters = [],[],[]
for m in movies:
    ids.append(m[0])
    titles.append(m[1])
    indexes = [i for i, ltr in enumerate(m[13]) if ltr == '/']
    localname =  m[13][indexes[-1]+1:]
    # posters.append([m[1],'/static/posters/%s' % localname])
    posters.append([m[1],m[13]])

with open('../static/movielist.json','w') as f:
    f.write(json.dumps(titles))


with open('../static/posterlist.json','w') as f:
    f.write(json.dumps(posters))

qry = "SELECT actor_name FROM actor_stats ORDER BY total DESC;"
actors = mydb.query(qry)
actors = [a[0] for a in actors]
with open('../static/actorlist.json','w') as f:
    f.write(json.dumps(actors[:5000]))

qry = "SELECT writer_name FROM writer_stats ORDER BY total DESC;"
writers = mydb.query(qry)
writers = [w[0] for w in writers]
with open('../static/writerlist.json','w') as f:
    f.write(json.dumps(writers))

qry = "SELECT director FROM director_stats ORDER BY total DESC;"
dirs = mydb.query(qry)
dirs = [w[0] for w in dirs]
with open('../static/directorlist.json','w') as f:
    f.write(json.dumps(dirs))