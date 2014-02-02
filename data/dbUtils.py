import mysql.connector
import json
import os
import sys
import requests
import urllib2
from bs4 import BeautifulSoup
from time import sleep

def get_user_pass():
    f=open('data/userpass.txt')
    for line in f:
        c=line.split(' ')
        user,pas = c[0].strip(),c[1].strip()
    f.close()
    print (user,pas)
    return (user,pas)

class db_access(object):
    """Access database
    """
    def __init__(self, db_name, usr, pwd=None):
        print 'in db_access'
        self.db_name = db_name
        self.db_url = "localhost"
        self.connect(usr, pwd)
        self.cursor = self.cnx.cursor()

    def query(self,qry):
        self.cursor.execute(qry)
        return [list(r) for r in self.cursor]

    def connect(self, usr, pwd=None):
        #try to connect to database
        try:
            self.cnx = mysql.connector.connect(user=usr, password=pwd, 
                database=self.db_name, host=self.db_url)
        except mysql.connector.Error as err:
            print(err)
            sys.exit(1)

    def close(self):
        """Disconnect from DB
        """
        self.cursor.close()
        self.cnx.close()



def cast_to_sql():
    mydb = db_access('rottentomatoes',os.environ['SQLUSER'],os.environ['SQLPASS'])
    files_list = os.listdir('json/casts')
    for f in files_list:
        fp = open('json/casts/'+f,'r')
        try:
            alldata = json.load(fp)
            imdb_id = alldata['imdb_id']
            
            for i in range(len(alldata['cast'])):
                c = alldata['cast'][i]
                print imdb_id

                qry = "SELECT * FROM casts where imdb_id = "+"\""+imdb_id+ \
                    "\" and actor_id = "+"\""+c['id']+"\";"

                mydb.cursor.execute(qry)
                casts = [m for m in mydb.cursor]
                if casts: continue

                params = ("\""+c['id']+"\"", "\""+imdb_id+"\"", "\""+alldata['rt_ind']+"\"", \
                    "\""+c['name']+"\"", "\""+c['characters'][0]+"\"", i)

                qry = "INSERT INTO casts "
                qry += "(actor_id, imdb_id, rt_id, nam, charac, billed_num) "
                qry += "VALUES (%s,  %s,  %s,  %s,  %s,  %d);" % params

                print qry;

                mydb.cursor.execute(qry)
                mydb.cnx.commit()

        except Exception,e:
            print e

        fp.close()

def download_casts():
    mydb = db_access('rottentomatoes',os.environ['SQLUSER'],os.environ['SQLPASS'])
    qry = ("""SELECT imdb_id,rt_id,title,cast FROM rt_info""")
    mydb.cursor.execute(qry)
    casts = [m for m in mydb.cursor]

    for [imdb_id,rt_id,title,cast_link] in casts:
        print title

        if os.path.isfile('json/casts/cast'+str(imdb_id)+'.json'):
            continue

        f = open('json/casts/cast'+str(imdb_id)+'.json','w')

        url = cast_link
        url += '?apikey=8pfmg4jgr8z3zcrpubh4v8pg'

        r = requests.get(url)
        js = r.json()
        d = dict(js)

        if 'error' in d and d['error'] == 'Gateway Timeout':
            continue
        elif 'error' in d:
            print 'error'

        tempdata = d['cast']
        data = {'imdb_id':imdb_id,'rt_ind':rt_id,'cast':tempdata}
        f.write(json.dumps(data))

        f.close()
        sleep(.2)


def rt_to_sql():
    mydb = db_access('rottentomatoes',os.environ['SQLUSER'],os.environ['SQLPASS'])

    files_list = os.listdir('json/movies')
    for f in files_list:
        imdb_id = f[9:f.index('.')]

        fp = open('json/movies/'+f,'r')
        js = json.loads(fp.read())
        fp.close

        for m in js:

            params = ("\""+imdb_id+"\"","\""+m['id']+"\"","\""+m['title']+"\"", \
                m['ratings']['critics_score'], \
                m['ratings']['audience_score'],"\""+m['links']['cast']+"\"", \
                "\""+m['posters']['thumbnail']+"\"", "\""+m['posters']['original']+"\"",)

            qry = "SELECT * FROM rt_info where imdb_id = "+"\""+imdb_id+"\""
            mydb.cursor.execute(qry)
            casts = [m for m in mydb.cursor]
            if casts: continue

            qry = "INSERT INTO rt_info "
            qry += "(imdb_id, rt_id, title, critic, aud, cast, thumb, orig) "
            qry += "VALUES (%s,  %s,  %s,  %d,  %d, %s, %s, %s);" % params
            print qry
            try:
                mydb.cursor.execute(qry)
                mydb.cnx.commit()
            except Exception,e:
                print e

def download_rt_info():
    mydb = db_access('rottentomatoes',os.environ['SQLUSER'], \
        os.environ['SQLPASS'])
    qry = "SELECT * FROM imdb_info"
    mydb.cursor.execute(qry)
    casts = [m for m in mydb.cursor]
    for c in casts: 
        #get movie info
        print c
        (imdb_id,title,yr,director,budget,gross) = c
        imdb_id = str(imdb_id)

        #check if we already have rt data
        if os.path.isfile('json/movies/moviepage'+imdb_id+'.json'):
            continue

        #query the rt API
        url = 'http://api.rottentomatoes.com/api/public/v1.0/movies.json'
        url += '?apikey=8pfmg4jgr8z3zcrpubh4v8pg&q='+str(title)
        url += '&page_limit=50'
        r = requests.get(url)
        js = r.json()
        #find the matching movie(s)
        ls = []
        print js
        if js and js['total'] > 0:
            for m in js['movies']:
                if m['year'] == yr:
                    print 'match'
                    ls.append(m)
            #save data locally if we have a match(es)
            if ls:
                f = open('json/movies/moviepage'+imdb_id+'.json','w')
                f.write(json.dumps(ls))
                f.close()

        sleep(.2)

def imdb_to_sql():
    mydb = db_access('rottentomatoes',os.environ['SQLUSER'],os.environ['SQLPASS'])
    fp = open('top_movies.txt','r')
    for line in fp:
        print line
        [title,yr,director,budget,gross,id] = line.split('\t')
        if gross == '-': gross = '-1'
        if budget == '-': budget = '-1'
        id = id.replace('\n','')

        qry = "SELECT * FROM imdb_info where imdb_id = "+"'"+id+"'"
        mydb.cursor.execute(qry)
        casts = [m for m in mydb.cursor]
        if casts:
            print 'pass'
            continue

        qry = "INSERT INTO rottentomatoes.imdb_info "
        qry += "(imdb_id, title, year, director, budget, gross) "
        qry += "VALUES (%s,  %s,  %d,  %s,  %d, %d);" % \
            ("\""+id+"\"","\""+title+"\"",int(yr),"\""+ \
            director+"\"",int(budget),int(gross))

        print qry
        try:
            mydb.cursor.execute(qry)
            mydb.cnx.commit()
        except Exception,e:
            print e

    fp.close()

def make_typeahead_list():
    mydb = db_access('rottentomatoes',os.environ['SQLUSER'],os.environ['SQLPASS'])
    qry = "SELECT title FROM imdb_info;"
    mydb.cursor.execute(qry)
    movieids = [m[0] for m in mydb.cursor]

    titles = []
    for m in movieids:
        titles.append(m)

    f = open('../static/movielist.json','w')
    f.write(json.dumps(titles))
    f.close()

def make_poster_list():
    mydb = db_access('rottentomatoes',os.environ['SQLUSER'],os.environ['SQLPASS'])
    qry = "SELECT title,orig FROM rt_info;"
    mydb.cursor.execute(qry)
    movieids = [m for m in mydb.cursor]
    titles = [m[0] for m in movieids]
    movieids = [m[1] for m in movieids]

    posters = []

    for i in range(len(movieids)):
        p,t = movieids[i],titles[i]
        print p
        indexes = [i for i, ltr in enumerate(p) if ltr == '/']
        localname =  p[indexes[-1]+1:]
        # posters.append([t,'static/posters/'+localname])
        posters.append([t,p])

    f = open('../static/posterlist.json','w')
    f.write(json.dumps(posters))
    f.close()

def writers_to_sql():
    mydb = db_access('rottentomatoes',os.environ['SQLUSER'],os.environ['SQLPASS'])
    files_list = os.listdir('json/writers')
    for f in files_list:
        fp = open('json/writers/'+f,'r')
        try:
            alldata = json.load(fp)
            imdb_id = alldata['imdb_id']
            
            for i in range(len(alldata['writers'])):
                c = alldata['writers'][i]
                writer_name,writer_id = c[0],c[1]

                print imdb_id

                qry = "SELECT * FROM writers where imdb_id = "+"\""+imdb_id+ \
                    "\" and writer_id = "+"\""+writer_id+"\";"

                casts = mydb.query(qry)
                if casts: continue

                params = ("\""+imdb_id+"\"", "\""+writer_id+"\"", "\""+writer_name+"\"")

                qry = "INSERT INTO writers "
                qry += "(imdb_id, writer_id, writer_name) "
                qry += "VALUES (%s,  %s,  %s);" % params

                print qry;

                mydb.cursor.execute(qry)
                mydb.cnx.commit()

        except Exception,e:
            print e

        fp.close()

def scrape_writers():
    mydb = db_access('rottentomatoes',os.environ['SQLUSER'],os.environ['SQLPASS'])
    qry = "SELECT imdb_id FROM imdb_info;"
    mydb.cursor.execute(qry)
    movieids = [m[0] for m in mydb.cursor]

    for m in movieids:
        print m
        url ='http://www.imdb.com/title/%s/fullcredits' % m
        print url
        try:
            #get the HTML of the page
            req = urllib2.Request(url)
            f = urllib2.urlopen(req)
            html = f.read()
        except:
            continue

        soup = BeautifulSoup(html)
        t = soup.findAll('a')

        movie_info = {'imdb_id':m, 'writers':[]}

        for td in t:
            if 'ttfc_fc_wr' in td['href']: 
                writer_id = td['href'][6:15]
                writer_name = td.contents[0].strip()
                if [writer_name,writer_id] not in movie_info['writers']:
                    movie_info['writers'].append([writer_name,writer_id])

        if movie_info['writers']:
            f = open('json/writers/'+m+'.json','w')
            f.write(json.dumps(movie_info))
            f.close()


#a function to scrape IMDB to get information for the top grossing movies
# of each year
def scrape_top_movies(vb=True):

    #a file for output
    ff = open('top_movies.txt','a')

    #iterate over years
    for yr in range(1984,2015):
        #only 50 movies are shown at a time
        for st in range(1,52,50):
            print yr,st

            #first scrape the top 100 movie page
            url = 'http://www.imdb.com/search/title?at=0&sort=boxoffice_gross_us,'
            url += 'desc&start='+str(st)+'&year='+str(yr)+','+str(yr)

            #get the HTML of the page
            req = urllib2.Request(url)
            f = urllib2.urlopen(req)
            html = f.read()
            soup = BeautifulSoup(html)
            td = soup.findAll('td',{'class':'title'})
            buds = soup.findAll('td',{'class':'sort_col'})
            ls = []

            #iterate over the table entry elements
            for i in range(len(td)):
                t,b = td[i],buds[i]

                #get the director and budget
                c = t.findAll('span',{'class':'credit'})
                dirname = c[0].contents[1].contents[0]
                budget = b.contents[0]

                #find the link with the movie title and id
                a = t.findNext('a')
                ls.append((a.contents[0],dirname,budget,a['href']))

            #iterate through the list
            for l in ls[-50:]:

                #get information scraped from the last page
                try:
                    title = str(l[0])
                    director = str(l[1])
                except Exception,e:
                    print e
                    continue

                gross = str(l[2]).replace('$','')
                if gross[-1] == 'K': fac = 1e3
                else: fac = 1e6

                if gross != '-':
                    gross = int(float(gross[:-1])*fac)

                budget = '-'
                id = l[3][7:-1]

                #go to the movie page, find the budget entry
                url = 'http://www.imdb.com/title/'+id
                req = urllib2.Request(url)
                f = urllib2.urlopen(req)
                html = f.read()
                soup = BeautifulSoup(html)
                txts= soup.findAll('div',{'class':'txt-block'})
                for t in txts:
                    a = t.findAll('h4',{'class':'inline'})
                    if a and a[0].contents[0] == 'Budget:':
                        budget = t.contents[2].replace('\n','').replace(',','') \
                            .replace(' ','').replace('\t','')[1:]

                #catch ascii error from strange characters
                try:
                    #console logging
                    if vb:
                        print title+'\t'+str(yr)+'\t'+director+'\t'+budget+'\t'+ \
                        str(gross)+'\t'+id+'\n'

                    #write the information to a text file
                    ff.write(title+'\t'+str(yr)+'\t'+director+'\t'+budget+'\t'+ \
                        str(gross)+'\t'+id+'\n')

                except Exception,e:
                    print e

    ff.close()



def cast2_to_sql():
    mydb = db_access('rottentomatoes',os.environ['SQLUSER'],os.environ['SQLPASS'])
    files_list = os.listdir('json/actors2')
    for f in files_list:
        fp = open('json/actors2/'+f,'r')
        try:
            alldata = json.load(fp)
            imdb_id = alldata['imdb_id']
            print imdb_id, alldata['actors']

            for i in range(len(alldata['actors'])):
                c = alldata['actors'][i]
                print c
                idnum = c[0]
                name = c[1]
                print imdb_id

                qry = "SELECT * FROM actors where imdb_id = "+"\""+imdb_id+ \
                    "\" and actor_id = "+"\""+idnum+"\";"

                mydb.cursor.execute(qry)
                casts = [m for m in mydb.cursor]
                if casts: continue

                params = ("\""+imdb_id+"\"", "\""+idnum+"\"", "\""+name+"\"", i)

                qry = "INSERT INTO actors "
                qry += "(imdb_id, actor_id, actor_name, billed_num) "
                qry += "VALUES (%s,  %s,  %s, %d);" % params

                print qry;

                mydb.cursor.execute(qry)
                mydb.cnx.commit()

        except Exception,e:
            print e

        fp.close()

def scrape_actors():
    mydb = db_access('rottentomatoes',os.environ['SQLUSER'],os.environ['SQLPASS'])
    qry = "SELECT imdb_id FROM imdb_info;"
    mydb.cursor.execute(qry)
    movieids = [m[0] for m in mydb.cursor]

    for m in movieids:
        print m
        url ='http://www.imdb.com/title/%s/fullcredits' % m
        print url
        try:
            #get the HTML of the page
            req = urllib2.Request(url)
            f = urllib2.urlopen(req)
            html = f.read()
        except:
            continue

        soup = BeautifulSoup(html)
        t = soup.findAll(attrs={'class':'itemprop'})
        actors = []

        cnt = 0
        for item in t:
            cnt += 1
            if cnt % 2 == 0: continue
            name = item.text.strip()
            idnum = item.findAll('a')[0]['href'][6:15]
            print name,idnum
            if name not in actors:
                actors.append([idnum,name])
            if len(actors) >= 10: break

        print actors

        movie_info = {'imdb_id':m, 'actors':actors}

        if movie_info['actors']:
            f = open('json/actors2/'+m+'.json','w')
            f.write(json.dumps(movie_info))
            f.close()
