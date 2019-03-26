import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from preprocessing import create_df
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from event_api_client import EventAPIClient
from model import MyModel

import pymongo
mc = pymongo.MongoClient(host = 'localhost', port=27017

def create_label(row):
    if 'fraud' in row['acct_type']:
        return True
    else:
        return False


def get_data(datafile):

    df = pd.read_json(datafile)
    df['fraud'] = df.apply(create_label, axis=1)
    df = pd.get_dummies(df, columns=['currency', 'country'])
    df.drop(df.select_dtypes(['object']), inplace=True, axis=1)

    y = df['fraud']
    X = df.drop(['fraud', 'delivery_method', 'event_published',
    'has_header', 'org_facebook', 'org_twitter', 'sale_duration',
    'venue_latitude', 'venue_longitude'], axis=1)
    return X, y

def clean_raw_data():

    df = pd.DataFrame(list(db['raw_collections'].find()))
    #df = pd.get_dummies(df, columns=['currency', 'country'])
    df.drop(df.select_dtypes(['object']), inplace=True, axis=1)
    X = df.drop(['delivery_method', 'event_published','org_facebook',
                 'org_twitter', 'sale_duration','venue_latitude', 'venue_longitude'], axis=1)
    return X

if __name__ == '__main__':
    X, y = get_data('data/data.json')
    model = MyModel()
    model.fit(X, y)
    with open('model.pkl', 'wb') as f:
        # Write the model to a file.
        pickle.dump(model, f)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    probs = model.predict_proba(x_test)

    #DATA BASE
    db = mc['events']
    raw_collections = db['row']

    client = EventAPIClient()
    client.collect()


    #

    ## Taking raw data and cleaning, then taking predictions.
    ## Putting into new database for flask.
    X_test = clean_raw_data()
    probs = model.predict_proba(X_test)
