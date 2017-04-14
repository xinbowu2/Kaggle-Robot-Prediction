import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn import preprocessing
import features_col as ft

from my_adaboost import my_adaboost

print "Reading features data..."
features = pd.read_csv('data/features.csv', index_col=1)
print "Done."

# Fixing Missing Values
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """
        Impute missing values.
        Columns of dtype object are imputed with the most frequent value
        in column.
        Columns of other types are imputed with mean of column.
        ** This technique got good results with previous classification, so it persists here
        ** I expect that the mean values we are replacing with belong to the human classification
        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

fixed_features = DataFrameImputer().fit_transform(features)
# save fixed features, not necessary
fixed_features.to_csv('../../data/fixed_features.csv', sep=',')

# Encode String Values so that random forest classifier can be applied
le = preprocessing.LabelEncoder()
le.fit(ft.country)
fixed_features['most_common_country'] = le.transform(fixed_features['most_common_country'])

# spilt test and train data from features.csv
test = fixed_features[fixed_features['outcome'] == -1]
train = fixed_features[fixed_features['outcome'] != -1]


def run_my_adaboost(train_input, test_input):
    clf = my_adaboost(100)
    clf.fit(train_input.drop(['outcome'], axis=1).values, train['outcome'].values)
    return clf.predict_proba(test_input.drop(['outcome'], axis=1).values)

# execute classification using my_adaboost
r = run_my_adaboost(train, test)
# save non-normalized results
np.savetxt('results/mySubmission_raw7.csv', r, delimiter=',')
# execute MinMax normalization of results (haven't tried with other normalization techniques)
r_norm = preprocessing.MinMaxScaler().fit_transform(r)
# save normalized results
np.savetxt('results/minMax_submission7.csv', r_norm, delimiter=',')

# add results to DataFrame (throws warning, but works just fine)
test['outcome'] = r_norm.T
# change column name from 'outcome' to 'prediction'
test = test.rename(columns={'outcome': 'prediction'})
# write result to csv
test.to_csv('results/mySubmission_final7.csv', sep=',', columns = ['prediction'])
