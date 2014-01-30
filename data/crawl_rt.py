import requests
import json
from time import sleep
import re
import os

#go through the movielens list to get a list of movie titles
movies = []
movielist = open('movies.dat','r')
pat = re.compile(' \(\d\d\d\d\)')
for line in movielist:
    c = line.split('::')
    match = pat.search(c[1])
    if match:
        movies.append(c[1][:match.start()])

#iterathe through the titles, querying RT for data
start = 508
end = len(movies)/2
i = start

while i < end:
    print i
    title = movies[i]

    f = open('json/moviepage'+str(i)+'.json','w')


    url = 'http://api.rottentomatoes.com/api/public/v1.0/movies.json'
    url += '?apikey=8pfmg4jgr8z3zcrpubh4v8pg&q='+title
    url += '&page_limit=50'

    r = requests.get(url)
    js = r.json()
    d = dict(js)
    if 'error' in d and d['error'] == 'Gateway Timeout':
        continue

    if d['total'] > 0:
        f.write(json.dumps(js['movies']))

    f.close()
    i += 1
    sleep(.2)
