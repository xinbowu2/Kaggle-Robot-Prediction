import pandas as pd
import numpy as np

import math
import random

from sklearn.tree import DecisionTreeClassifier

class MyExtraTree:

    def _init_(self, n_estimators):
        self.num_tree = n_estimators


    def set_max_features(self, train):
        self.num_samples, self.num_features = train.shape
        max_features = log2(num_features)

    def generate_train_for_tree(self, train, target):
                sample_indices = np.ones(self.num_samples)
                
		features_indices = random.sample(range(self.n_features), self.max_features) 

		tree_train = train.iloc[samples_indices,features_indices]
		tree_target = target.iloc[samples_indices]

		return (tree_train, tree_target, features_indices)
          
    def fit (self, train, label):
        self.set_max_features(train)

        
        for i in range(self.num_tree):
            self.trees[i] = DecisionTreeClassifier()
            (tree_train, tree_target, self.tree_features[i]) = self.generate_train_for_tree(train, target)
             self.trees[i].fit(tree_train, tree_target)
    return self


    def predict_proba(self, test):
		predictions = pd.DataFrame()

		for i in range(self.n_trees):

			tree_test = test.iloc[:, self.tree_features[i]]
			prediction = self.trees[i].predict_proba(tree_test)
			
			df_pdct = pd.DataFrame(prediction)
			predictions = predictions.add(df_pdct, fill_value=0)

		average = lambda x: x/self.n_trees
		return predictions.apply(average).as_matrix()
    
