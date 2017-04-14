import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing   
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

import features_col as ft


# use the first column (bidder_id) in the csv as the row index
features = pd.read_csv('data/features.csv', index_col = 1)

# Fixing Missing Values
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

fixed_features = DataFrameImputer().fit_transform(features)
fixed_features.to_csv('data/fixed_features.csv', sep=',')


# Encode String Values so that random forest classifier can be applied
le = preprocessing.LabelEncoder()
le.fit(ft.country)
fixed_features['most_common_country'] = le.transform(fixed_features['most_common_country'])


# spilt test and train data from features.csv
test = fixed_features[fixed_features['outcome']==-1]
train = fixed_features[fixed_features['outcome']!=-1]

# apply random forest classifier
rf = RandomForestClassifier(n_estimators=500)
adB = AdaBoostClassifier(n_estimators = 100)
SVC = SVC(kernel='rbf', probability=True) 

eclf = VotingClassifier(estimators=[('adB', adB), ('rf', rf),('SVC', SVC)], voting='soft', weights=[2,2,1])
params = {'adB__n_estimators': [100, 200], 'rf__n_estimators': [500, 1000],}

grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
grid = grid.fit(train.drop(['outcome'], axis=1), train['outcome'])


test['outcome'] = grid.predict_proba(test.drop(['outcome'], axis=1))[:,1]                                     
# change column name from 'outcome' to 'prediction'

test = test.rename(columns = {'outcome':'prediction'})

# write result to csv
test.to_csv('data/mySubmission.csv', sep=',', columns = ['prediction'])


