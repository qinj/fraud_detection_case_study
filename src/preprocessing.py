import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def create_label(row):
    if 'fraud' in row['acct_type']:
        return True
    return False

def create_df():
    df = pd.read_json('../data/data.json')
    df['fraud'] = df.apply(create_label, axis=1)
    df = pd.get_dummies(df, columns=['currency', 'country'])
    df.drop(df.select_dtypes(['object']), inplace=True, axis=1)
    return df
