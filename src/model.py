import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle
import pymongo
from collections import Counter
import json

def create_label(row):
    if 'fraud' in row['acct_type']:
        return True
    else:
        return False



def create_label(lst):
    label_list = []
    for item in lst:
        if item == 'premium':
            label_list.append(0)
        elif item in ['fraudster','fraudster_event', 'fraudster_att']:
            label_list.append(1)
        else:
            label_list.append(2)
    return label_list

def get_data(datafile):
    mc = pymongo.MongoClient(host='localhost', port=27017)
    # df = pd.read_json(datafile)
# MONGO
    db = mc['event_shiny']
    trans_coll = db['transactions']
    with open('../data/data.json') as f:
        file_data = json.load(f)
    deleted = trans_coll.delete_many({})
    trans_coll.insert_many(file_data)
    cursor = trans_coll.find()
    for doc in cursor:
        if 'previous_payouts' in doc and type(doc['previous_payouts']) is list:
            previous_payouts_count = len(doc['previous_payouts'])
        else:
            previous_payouts_count = 0
        _id = doc['_id']
        trans_coll.update_one(
            {'_id': _id},
            {'$set': {'previous_payouts_count': previous_payouts_count}})
    cursor = trans_coll.find()
    data = pd.DataFrame(list(cursor))
    label_list = create_label(data['acct_type'].values)
    feature_union = data[['previous_payouts_count', 'user_age', 'body_length', 'has_analytics', 'sale_duration']].copy()
    feature_union['sale_duration'] = feature_union['sale_duration'].fillna(feature_union['sale_duration'].mean())
    y = np.array(label_list)
    X = feature_union.values
    deleted = trans_coll.delete_many({})

    mc.close()
    return X, y
#



    # df['fraud'] = df.apply(create_label, axis=1)
    # df = pd.get_dummies(df, columns=['currency', 'country'])
    # df.drop(df.select_dtypes(['object']), inplace=True, axis=1)

    # y = df['fraud']
    # X = df.drop(['fraud', 'delivery_method', 'event_published',
    # 'has_header', 'org_facebook', 'org_twitter', 'sale_duration',
    # 'venue_latitude', 'venue_longitude'], axis=1)
    # return X, y


class MyModel():
    def __init__(self):
        pass

    def fit(self, X, y):
        self.rf = RandomForestClassifier(n_estimators = 100)
        self.rf.fit(X, y)
        # self.lr = LogisticRegression(penalty='l2', solver='lbfgs')
        # return self.lr.fit(X, y)


    def predict_proba(self, X_test):
        return self.rf.predict_proba(X_test)
        # return self.lr.predict(X_test)

if __name__ == '__main__':
    X, y = get_data('../data/data.json')
    model = MyModel()
    model.fit(X, y)
    with open('model.pkl', 'wb') as f:
        # Write the model to a file.
        pickle.dump(model, f)
