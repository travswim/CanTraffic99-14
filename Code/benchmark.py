#Import stuff
import progressbar
from time import time
from tqdm import *
import pandas as pd
import random
import numpy as np

filename = "NCDB_1999_to_2014.csv"
df = readFile(filename)

print "Data read successfully"

#define the fatalities/non-fatalities
fatal = (df['C_SEV'] == 1).sum()
nonFatal = (df['C_SEV'] == 2).sum()
total_accidents = len(df['C_SEV'])

#Print some stuff
print "Total number of fatalities: {}".format(fatal)

print "Total number of non-fatalities: {}".format(nonFatal)

print "Total number of Accidents: {}".format(total_accidents)

feature_cols = list(df.drop('C_SEV', 1))
#Find the target columns
target_col = 'C_SEV'
del fatal, nonFatal
try:
    X_all = df[feature_cols]
    y_all = df['C_SEV']
    # print "Target: {}".format(y_all)
    # print "Features: {}".format(feature_cols)

    print "Successfully separated features and target"
    print X_all.shape
except:
    print "Failed to separate features and target\n"

del df
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_all, y_all, stratify=y_all, 
                                                    test_size=0.24, random_state=42)

print "Train set 'fatal' pct = {:.2f}%".format(100 * (y_train == 1).mean())
print "Test  set 'fatal' pct = {:.2f}%".format(100 * (y_test == 1).mean())
# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])                                

# Scoring Metrics
from sklearn.metrics import f1_score
def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
   
    start = time()
    y_pred = clf.predict(features)
    end = time()
  
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label=1)
# , accuracy_score(target.values, y_pred), matthews_corrcoef(target.values, y_pred)]


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))
    # print "Accuracy score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)[1])
    # print "Matthews CC for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)[2])
# Now we need some classifiers, using the scikit-learn algorithm cheat-shee:
# Import classifiers:
# TODO: Import the three supervised learning models from sklearn

# TODO: Initialize the three models
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

print "\n{}: \n".format(clf.__class__.__name__)
train_predict(clf, X_train, y_train, X_test, y_test)
