from flask import Flask, request, render_template, jsonify
from flask_pymongo import PyMongo
from model import MyModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import pymongo

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__, static_url_path="")

app.config["MONGO_URI"] = "mongodb://localhost:27017/events"
app.config["MONGO_DBNAME"] = "events"

mongo = PyMongo(app)

def clean_raw_data():
    row = mongo.db.row
    df = pd.DataFrame(list(row.find()))
    if df is not None:
        #df = pd.get_dummies(df, columns=['currency', 'country'])
        l = []
        for i, row in df.iterrows():
            l.append(len(row['previous_payouts']))
        df['previous_payouts_count'] = pd.Series(l)
        X = df[['previous_payouts_count', 'user_age', 'body_length', 'has_analytics', 'sale_duration']].copy()
        # df.drop(df.select_dtypes(['object']), inplace=True, axis=1)
        # X = df.drop(['delivery_method', 'event_published','org_facebook',
        #             'org_twitter', 'sale_duration','venue_latitude', 'venue_longitude'], axis=1)

        return X
    else:
        return None

@app.route('/score', methods=['GET'])
def display_predictions():
    ## Taking raw data and cleaning, then taking predictions.
    ## Putting into new database for flask.
    X_test = clean_raw_data()

    probs = model.predict_proba(X_test)
    prob_list = []
    for i in probs:
        prob_list.append(i)
    X_test['probs'] = prob_list
    return X_test.to_html()

@app.route('/')
def index():
   """Return the main page."""
   return render_template('index.html')

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=8080, debug=True)
