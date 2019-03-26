from preprocessing import create_df
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

df = create_df()

y = df['fraud']

# contains nans
df = df.drop(['fraud', 'delivery_method', 'event_published',
'has_header', 'org_facebook', 'org_twitter', 'sale_duration',
'venue_latitude', 'venue_longitude'], axis=1)

# need to scale data
X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

lr = LogisticRegression(penalty='l2', solver='lbfgs')
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

auroc = roc_auc_score(y_test, y_pred)
print(auroc)
