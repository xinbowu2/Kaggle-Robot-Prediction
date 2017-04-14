
import pandas as pd
import numpy as np

import math
import random

from sklearn.tree import DecisionTreeClassifier

class MyRandomForest:
	''' My own random forest implementation based on sklearn's decision tree'''
	def __init__(self, n_estimators):
		self.n_trees = n_estimators

		self.trees = [None] * self.n_trees
		self.tree_features = [None] * self.n_trees

	def set_numbers(self, train):
		self.n_samples, self.n_features = train.shape
		self.max_features = int(math.sqrt(self.n_features))

	def generate_random_samples_indices(self):
		result = []
		for i in range(self.n_samples):
			result.append(random.randint(0, self.n_samples-1))
		return result

	def generate_train_for_tree(self, train, target):
		samples_indices = self.generate_random_samples_indices() # with replacement
		

		features_indices = random.sample(range(self.n_features), self.max_features)
		# without replacement

		tree_train = train.iloc[samples_indices,features_indices]
		tree_target = target.iloc[samples_indices]

		return (tree_train, tree_target, features_indices)

	def fit(self, train, target):
		self.set_numbers(train)

		for i in range(self.n_trees):

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


