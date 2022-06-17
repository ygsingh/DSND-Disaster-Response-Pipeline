import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse_table', engine)
#df = df[df.columns[4:]]
# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    
    
    # There are three genre, lets plot the distribution of categories in 
    # each genre 
    # genre name == 'Direct'
    direct = df[df.genre == 'direct']
    direct = direct[direct.columns[4:]]
    direct_counts = (direct.mean()*direct.shape[0]).sort_values(ascending=False)
    direct_names = list(direct_counts.index)
    
    # genre_name == 'news'
    news = df[df.genre == 'news']
    news = news[news.columns[4:]]
    news_counts = (news.mean()*news.shape[0]).sort_values(ascending=False)
    news_names = list(news_counts.index)
    
    # genre_name == 'social'
    social = df[df.genre == 'social']
    social = social[social.columns[4:]]
    social_counts = (social.mean()*social.shape[0]).sort_values(ascending=False)
    social_names = list(social_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # Graph 1 provided with template
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # Graph 2
        {
            'data':[
                Bar(
                    x = direct_names,
                    y = direct_counts
                )
            ],
            
            'layout':{
                'title':'Distribution of categories in direct genre',
                'yaxis':{'title':'Count'},
                'xaxis':{'title':'Categories'}            
            }
        },
        # Graph 3
        {
            'data':[
                Bar(
                    x = news_names,
                    y = news_counts
                )
            ],
            
            'layout':{
                'title':'Distribution of categories in news genre',
                'yaxis':{'title':'Count'},
                'xaxis':{'title':'Categories'}            
            }
        },
        # Graph 4
        {
            'data':[
                Bar(
                    x = social_names,
                    y = social_counts
                )
            ],
            
            'layout':{
                'title':'Distribution of categories in social genre',
                'yaxis':{'title':'Count'},
                'xaxis':{'title':'Categories'}            
            }
        }

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()