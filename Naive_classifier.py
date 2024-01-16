#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class MultinomialNB():
    def fit(self, X_train, y_train):
        self.m, self.n = X_train.shape
        self.classes = np.unique(y_train)
        self.n_classes = len(self.classes)

        # init: Prior & Likelihood
        self.priors = np.zeros(self.n_classes)
        self.likelihoods = np.zeros((self.n_classes, self.n))

        # Get Prior and Likelihood
        for idx, c in enumerate(self.classes):
            X_train_c = X_train[c == y_train]
            self.priors[idx] = X_train_c.shape[0] / self.m   # P(C)
            self.likelihoods[idx, :] = ((X_train_c.sum(axis=0)) + 1) / (np.sum(X_train_c.sum(axis=0) + 1))  # P(F1, ..., FN | C)


    def predict(self, X_test):
        return [self._predict(x_test) for x_test in X_test]

    def _predict(self, x_test):
        # Calculate posterior for each class
        posteriors = []
        for idx, _ in enumerate(self.classes):
            prior_c = np.log(self.priors[idx])
            likelihoods_c = self.calc_likelihood(self.likelihoods[idx,:], x_test)
            posteriors_c = np.sum(likelihoods_c) + prior_c
            posteriors.append(posteriors_c)

        return self.classes[np.argmax(posteriors)]

    def calc_likelihood(self, cls_likeli, x_test):
        return np.log(cls_likeli) * x_test.toarray().flatten()

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.sum(np.array(y_pred) == np.array(y_test))/len(y_test)


class BernoulliNB():
    def fit(self, X_train, y_train):
        self.m, self.n = X_train.shape
        self.classes = np.unique(y_train)
        self.n_classes = len(self.classes)

        # init: Prior & Likelihood
        self.priors = np.zeros(self.n_classes)
        self.likelihoods = np.zeros((self.n_classes, self.n))

        # Get Prior and Likelihood
        for idx, c in enumerate(self.classes):
            X_train_c = X_train[c == y_train]
            self.priors[idx] = X_train_c.shape[0] / self.m   # P(C)
            n_doc = X_train_c.shape[0] + 2
            self.likelihoods[idx, :] = ((X_train_c.sum(axis=0)) + 1) / (n_doc)  # P(F1, ..., FN | C)


    def predict(self, X_test):
        return [self._predict(x_test) for x_test in X_test]

    def _predict(self, x_test):
        # Calculate posterior for each class
        posteriors = []
        for idx, _ in enumerate(self.classes):
            prior_c = np.log(self.priors[idx])
            likelihoods_c = self.calc_likelihood(self.likelihoods[idx,:], x_test)
            posteriors_c = np.sum(likelihoods_c) + prior_c
            posteriors.append(posteriors_c)

        return self.classes[np.argmax(posteriors)]

    def calc_likelihood(self, cls_likeli, x_test):
        return np.log(cls_likeli) * x_test.toarray().flatten() + np.log(1 - cls_likeli) * np.abs(1 - x_test.toarray().flatten())


    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.sum(np.array(y_pred) == np.array(y_test))/len(y_test)
        

def gen_train_test_data():
    test_ratio = .3 
    df = pandas.read_csv('./yelp_labelled.txt', sep='\t', header=None, encoding='utf-8')
    count_vect = CountVectorizer()
    X = count_vect.fit_transform(df[0])
    y = df[1].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=0)
    return X_train, X_test, y_train, y_test


def multinomial_nb(X_train, X_test, y_train, y_test):
    # TODO: fill this function
    # train by X_train and y_train
    # report the predicting accuracy for both the training and the test data 
    print("Multinomial:")
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    print("Train: %.4f" % clf.score(X_train, y_train))
    print("Test: %.4f" % clf.score(X_test, y_test))


def bernoulli_nb(X_train, X_test, y_train, y_test):
    # TODO: fill this function
    # train by X_train and y_train
    # report the predicting accuracy for both the training and the test data 
    print("Bernoulli:")
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    print("Train: %.4f" % clf.score(X_train, y_train))
    print("Test: %.4f" % clf.score(X_test, y_test))


def main(argv):
    X_train, X_test, y_train, y_test = gen_train_test_data()

    multinomial_nb(X_train, X_test, y_train, y_test)
    bernoulli_nb(X_train, X_test, y_train, y_test) 

if __name__ == "__main__":
    main(sys.argv)


