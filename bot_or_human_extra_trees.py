
import pandas as pd
import numpy as np
from my_extra_tree import MyExtraTree

from sklearn import preprocessing   
from sklearn.base import TransformerMixin

import features_col as ft


'''  use the first column (bidder_id) in the csv as the row index''' 
features = pd.read_csv('data/features.csv', index_col = 1)

'''  Fixing Missing Values''' 
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



le = preprocessing.LabelEncoder()
le.fit(ft.country)
fixed_features['most_common_country'] = le.transform(fixed_features['most_common_country'])



'''  spilt test and train data from features.csv''' 
test = fixed_features[fixed_features['outcome']==-1]
train = fixed_features[fixed_features['outcome']!=-1]

''' initialize the extra tree classifier with parameter = 500 trees''' 
et = MyExtraTree(n_estimators = 500);


''' train the extra tree classifier''' 
et = et.fit( train.drop(['outcome'], axis=1), train['outcome'])


''' generate probabilities of predictions by the extra tree classifier''' 
test['outcome'] = et.predict_proba(test.drop(['outcome'], axis=1))[:,1]

test = test.rename(columns = {'outcome':'prediction'})

''' write result to csv''' 
test.to_csv('data/mySubmission.csv', sep=',', columns = ['prediction'])
