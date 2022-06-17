# DSND Disaster Response Pipeline Project

### Project Description 

In this project, we have build a model for an API that classifies disaster messages. A skeleton code was provided by [Udacity](https://learn.udacity.com/nanodegrees/nd025/parts/cd0018/lessons/ea367f74-3d5a-42b1-92a3-d3d3734fd369/concepts/d7e645c3-a521-4214-8bd5-30e7137365cc) and the disaster data from [Appen](https://appen.com) (formally Figure 8). 

Using the data and the skeleton code, we created a machine learning pipeline to categorize disaster events. By doing so these messages can be send to an appropriate relief agency. 

This project included a web app where an emergeny worker can input a new message and get classification results in several categories. There are some training dataset visualizations on the web app.

### File structure 

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
