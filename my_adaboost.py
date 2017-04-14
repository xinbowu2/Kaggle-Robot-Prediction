__author__ = 'wbarbour'

from sklearn.tree import DecisionTreeClassifier
import random
import math
import numpy as np
np.set_printoptions(threshold=np.nan)


class my_adaboost:
    """
    Self-implementation of Adaboost ensemble classifier, optimized for sparse data
    Written for CS 412: Data Mining at the University of Illinois at Urbana-Champaign
    Author: William Barbour

    Part of a final group project: one of three full algorithm implementations
    Direct application was Kaggle's "Facebook Recruiting IV: Human or Robot?" competition
    Uses, as base/weak classifier, sklearn's DecisionTreeClassifier
    Also uses as utilities (non-functioning elements) Numpy and Pandas packages
    """
    def __init__(self, n_estimators=40, threshold=0.5):
        self.n_weak_learners = n_estimators
        self.weak_learners = [None] * self.n_weak_learners
        self.sample_weights = [None] * self.n_weak_learners
        self.errors = [None] * self.n_weak_learners
        self.score_threshold = threshold
        self.alphas = [None] * self.n_weak_learners

    def divide_train_sample(self, train_data, labels, sample_proportion):
        """
        Divides training data randomly for base/weak classifier construction
        Parameters
        ----------
        train_data: full training data set passed to fit(...)
        labels: full ground truth of classification for training data
        sample_proportion: sample_prop.*len(train_data) returned as train_sub, truth_sub, w_sub
                remainder (1-sample_prop.)*len(train_data) returned as remainder_test_sub, rem._truth_sub, r_w_sub, r_i
        self.sample_weights: returns subsets of sample weights (easier to do here than later)

        Returns
        -------
        Groupings mirror each other in original index
        {
            train_sub: randomly selected subset (fully featured) of training data - with replacement
            truth_sub: corresponding (by index) subset of truth labels
            w_sub: corresponding subset of sample weights
        }
        {
            remainder_test_sub: remainder of training data after train_sub is taken
            remainder_truth_sub: corresponding (by index) subset of truth labels to evaluate weak classifier
            r_w_sub: corresponding subset of sample weights
            r_i: list of indexes that remainder subset consists of
        }

        """
        p = int(round(train_data.__len__() * sample_proportion))
        p_i = random.sample(range(train_data.__len__()), p)
        r_i = []
        train_sub = []
        truth_sub = []
        w_sub = []
        remainder_test_sub = []
        remainder_truth_sub = []
        r_w_sub = []
        for i in range(train_data.__len__()):
            if i in p_i:
                train_sub.append(train_data[i])
                truth_sub.append(labels[i])
                w_sub.append(self.sample_weights[i])
            else:
                r_i.append(i)
                remainder_test_sub.append(train_data[i])
                remainder_truth_sub.append(labels[i])
                r_w_sub.append(self.sample_weights[i])
        train_sub = np.array(train_sub)
        truth_sub = np.array(truth_sub)
        w_sub = np.array(w_sub)
        remainder_test_sub = np.array(remainder_test_sub)
        remainder_truth_sub = np.array(remainder_truth_sub)
        r_w_sub = np.array(r_w_sub)
        r_i = np.array(r_i)
        return train_sub, truth_sub, w_sub, remainder_test_sub, remainder_truth_sub, r_w_sub, r_i

    def fit(self, train_data, labels):
        """
        Trains Adaboost ensemble classifier model, using n_estimators specified in constructor
        Needs to be called before call to predict_proba(test_data) can occur
        Parameters
        ----------
        train_data: fully featured training data for model construction
        labels: ground truth class labels, aligned by row with train_data: binary {0, 1}
                slight modification would be needed to change binary {0, 1} to {-1, 1}

        Returns
        -------
        None
        """
        self.sample_weights = (1. / train_data.__len__()) * np.ones(train_data.__len__())
        print "Fitting classifier on", self.n_weak_learners, "weak learners..."
        for i in range(self.n_weak_learners):
            print "Fitting weak learner", i
            wl = DecisionTreeClassifier()
            (train_sub, truth_sub, w_sub, rem_test_sub, rem_truth_sub, r_w_sub, r_i) = \
                self.divide_train_sample(train_data, labels, 0.4)
            print "Training set divided."
            wl.fit(train_sub, truth_sub, sample_weight=w_sub)
            print "Decision tree fitted"
            wl_pred = wl.predict_proba(rem_test_sub)[:, 1]
            epsilon = float(sum(0.5*r_w_sub*abs(rem_truth_sub-wl_pred))/sum(r_w_sub))
            print "epsilon =", epsilon
            if epsilon <= 0.0:
                epsilon = 0.00001
            alpha = 0.5*math.log((1-epsilon)/epsilon)

            # compute h = binary{-1, 1} for {incorrect, correct}
            h = np.zeros(wl_pred.__len__())
            for j in range(h.__len__()):
                if wl_pred[j] == rem_truth_sub[j]:
                    h[j] = 1
                else:
                    h[j] = -1

            # compute new sample weighting
            r_w_sub *= np.exp(-alpha*h)
            for rem_i in range(r_i.__len__()):
                self.sample_weights[r_i[rem_i]] = r_w_sub[rem_i]
            # normalize weights
            self.sample_weights /= sum(self.sample_weights)
            self.alphas[i] = alpha
            self.weak_learners[i] = wl
        print "Done."

    def predict_proba(self, test_data):
        """
        After model has been trained , probabilities of classification can be evaluated
        Currently only works on binary classification {0, 1}, but can be changed by slight modification to fit method
        Output is not normalized, and will scale linearly with n_estimators because of summation of probabilities
        Simple MinMax normaliztion of output seems to be effective
        Parameters
        ----------
        test_data: data to evaluate for classification, mirroring feature vector of test data

        Returns
        -------
        non-normalized output of classification probability via numpy array of size [length(test_data)]
        """
        predictions = []
        print "Evaluating test set on each weak learner..."
        for i in range(self.n_weak_learners):
            print "Predicting on weak learner", i
            pred = self.weak_learners[i].predict_proba(test_data)[:, 1]
            predictions.append(pred.tolist())
        print "Done."

        final_prediction = []
        print "Summing predictions..."
        # iterate over test data set elements
        for i in range(predictions[0].__len__()):
            f = 0.
            # iterate over number of predictions
            for j in range(predictions.__len__()):
                f += (predictions[j][i] * self.alphas[j])
            final_prediction.append(f)
            # TODO: shift classification from (sign) to [0,1]
            # probabilities returned are not normalized
            # MinMax normalization occurs after initialization of class and call to method
        print "Done."
        assert final_prediction.__len__() == test_data.__len__()
        return np.array(final_prediction)


