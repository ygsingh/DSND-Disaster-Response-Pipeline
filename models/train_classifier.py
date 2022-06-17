# Import libraries 
import sys
import os
from sqlalchemy import create_engine

import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle

def load_data(database_filepath):
    '''
    This function loads table from the SQLite database, find X and Y 
    variables and the categories names.
    
    Input:
        database_filepath -> SQLite database filepath
        
    Output:
        X -> feature variable dataframe
        Y -> target variable dataframe
        category_names -> classificaton types or classes 
    '''
    
    # Load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","")+"_table"
    df = pd.read_sql_table(table_name,engine)
    
    # Clean data
    df = df.drop(['child_alone'],axis=1)
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    # Assign variables 
    X = df.message
    Y = df.iloc[:,4:]
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    '''
    This function converts the text messages into a list of lemmatized tokens
    
    Input:
        text -> string of message
        
    Output:
        tokens -> list of words lemmetized from the message string
    '''
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words("english")]
    
    return tokens


def build_model():
    '''
    This function builds the pipeline needed to train the ML model
    
    Input: 
    
    Output:
        pipeline
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Setting up the grip: Small space because it takes ages to run this and the resuting output
    # classifier is 900MB, cann't upload it to GutHub.
    parameters = {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2],
    
    }
    
    # Gridsearch to find optimal model 
    model = GridSearchCV(pipeline, param_grid=parameters, cv=2)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function evaluates the performace of the trained model.
    Performance matrices displayed are: precision, recall, f1-score and support
    
    Input:
        model -> This is the trained model
        X_test -> test feature variables in dataframe format
        Y_test -> test target variables in dataframe format
        category_names -> list of names of target variable classes
        
    Output:
        report printed out on console 
    '''
    
    Y_pred = model.predict(X_test)
    report = classification_report(Y_test,Y_pred,target_names=category_names)
    print(report)


def save_model(model, model_filepath):
    '''
    This function saves the model as a pickle file
    
    Input:
        model -> model variable/object to be saved
        model_filepath -> location at which model need to be saved 
    '''
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
        main()