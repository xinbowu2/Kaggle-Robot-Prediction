import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.base import TransformerMixin

import features_col as ft

from my_random_forest import MyRandomForest

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

#drop some columns
#fixed_features.drop(ft.addr, inplace=True, axis=1)
#fixed_features.drop(ft.pay_acc, inplace=True, axis=1)
#fixed_features.drop(ft.category, inplace=True, axis=1)

# spilt test and train data from features.csv
test = fixed_features[fixed_features['outcome']==-1]
train = fixed_features[fixed_features['outcome']!=-1]


def my_random_forest(train, test):
    clf = MyRandomForest(n_estimators=500)

    clf = clf.fit(  train.drop(['outcome'], axis=1), train['outcome']  )
    return clf.predict_proba(test.drop(['outcome'], axis=1))[:,1]

# cross-validation
def kfold(n, train):
    scores = []

    for i in range(0, n):
        
        k_fold = KFold(2013, n_folds=5)
        
        for train_indices, test_indices in k_fold:

            cv_train, cv_test = train.iloc[train_indices], train.iloc[test_indices]
            
            # Specify Classkfier here
            cv_test_prediction = my_random_forest(cv_train, cv_test) 
            
            scores.append(roc_auc_score(cv_test['outcome'], cv_test_prediction))
        print("The ", i+1,"th time k fold done")
    
    print("Cross Validation Avg Score: ", np.mean(scores))


test['outcome'] = my_random_forest(train, test)

# change column name from 'outcome' to 'prediction'
test = test.rename(columns = {'outcome':'prediction'})

# write result to csv
test.to_csv('data/mySubmission.csv', sep=',', columns = ['prediction'])

