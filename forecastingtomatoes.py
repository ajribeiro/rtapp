from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
import MySQLdb
import data.ml as ml
from werkzeug.datastructures import ImmutableMultiDict
from code import interact

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/recalc",methods=['POST', 'GET'])
def recalc():
    dd = request.json
    print dd['actors']

    ret = ml.regress_cast(dd['actors'],dd['writers'],dd['director'][0])

    print ret

    return jsonify({'result':int(round(ret))})

@app.route("/new_movie")
def new_movie():
    return render_template('new_movie.html')

@app.route("/search_movie")
def search_movie():
    title = request.args.get('title')
    ret = ml.regress_title(title)
    if ret == None:
        return 'Couldnt find the title'

    if ret['prediction'] > 59: rating = 1
    else: rating = 0

    if ret['actual'][0] > 59:
        acrat = 1
    else:
        acrat = 0

    if ret['actual'] > 59: afresh = 1
    else: afresh = 0


    castlist = ret['cast']
    # if len(castlist) > 10:
    #     castlist = castlist[:10]

    return render_template('result.html',title=title,
        prediction=int(round(ret['prediction'])), actual=ret['actual'][0], 
        thumb=ret['thumb'], poster=ret['poster'], acrat=acrat,
        rating=rating, cast=castlist, director=ret['director'],
        writers=ret['writers'])


if __name__ == "__main__":
    app.debug=True
    app.run(host='0.0.0.0')