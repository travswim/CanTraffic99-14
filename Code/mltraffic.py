""" Machine learning file for Canadian Traffic Data 1999-2014"""
#Import stuff
import progressbar
from time import time
from tqdm import *
import pandas as pd
import random
import numpy as np

def readFile(filename):
    df = pd.read_csv(filename)
    # Numbers
    fatal = (df['C_SEV'] == 1).sum()
    nonFatal = (df['C_SEV'] == 2).sum()
    total_accidents = len(df['C_SEV'])
    # Dataframes
    dfatal = df.loc[df['C_SEV'] == 1]
    dfnonfatal = df.loc[df['C_SEV'] == 2]

    # Randomly sample % of your dataframe
    fatal_frac = 0.2
    print("Using {}% of fatatalities".format(fatal_frac))
    nonfatal_frac = fatal*fatal_frac/nonFatal

    df_sample_fatal = dfatal.sample(frac=fatal_frac)
    df_sample_nonfatal = dfnonfatal.sample(frac=nonfatal_frac)
    print(len(df_sample_nonfatal))
    frames = [df_sample_fatal, df_sample_nonfatal]
    result = pd.concat(frames)
    return result.sample(frac=1)


filename = "NCDB_1999_to_2014.csv"
df = readFile(filename)
# df = pd.read_csv(filename)
print "Data read successfully"

#define the fatalities/non-fatalities
fatal = (df['C_SEV'] == 1).sum()
nonFatal = (df['C_SEV'] == 2).sum()
total_accidents = len(df['C_SEV'])

#Print some stuff
print "Total number of fatalities: {}".format(fatal)

print "Total number of non-fatalities: {}".format(nonFatal)

print "Total number of Accidents: {}".format(total_accidents)
# raw_input("Press Enter to continue...")
del fatal, nonFatal
#Find the feature columns
feature_cols = list(df.drop('C_SEV', 1))
#Find the target columns
target_col = 'C_SEV'
#Print some more stuff
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)




# Show the feature information by printing the first five rows
# print "\nFeature values:"
# print X_all.head()

# Data Formatiing: Lets see if the data is uniform

# This function is to replace wrong input data with the correct ones
# (it would have been easier if the data had been entered correctly :/)

# Print list of items for a given feature
def data(feature):
    print list(df[feature].unique())


# Find most common item
def most_common(lst):
    return max(set(lst), key=lst.count)

# Replace Null values with most common element
def is_null(feature):
    df[feature] = df[feature].replace('nan', most_common(list(df[feature])))
    df[feature] = df[feature].replace('', most_common(list(df[feature])))
    df[feature] = df[feature].replace('U', most_common(list(df[feature])))
    df[feature] = df[feature].replace('UU', most_common(list(df[feature])))
    df[feature] = df[feature].replace('X', most_common(list(df[feature])))
    df[feature] = df[feature].replace('XX', most_common(list(df[feature])))
    df[feature] = df[feature].replace('Q', most_common(list(df[feature])))
    df[feature] = df[feature].replace('QQ', most_common(list(df[feature])))
# Find the data type of an item
def data_type(feature):
    # for feature in feature_cols:
    tp = []
    for item in list(df[feature].unique()):
        tp.append(type(item))
    print feature
    print tp

# Check and replace items not formated as strings
def not_string(feature):
    for item in df[feature].unique():
        if type(item) is not str:
            df[feature] = df[feature].replace(item, str(item))


# Check if all items in a list are the same length
def length(feature):
    ln = []
    for item in list(df[feature].unique()):
        ln.append(len(str(item)))
    # Check if all items in list are the same
    return all(x == ln[0] for x in ln)
# print "C_VEHS"
# data("C_VEHS")

## As we can see, not only is the data of mixed types (int and string, supposed
## to be stings) its also contains duplicates ('1' and '01 are equivalent') and 'nan'.


# This section calls various functions above and adds '0' to '1'
pbar = tqdm(feature_cols)
for feature in tqdm(feature_cols):
    pbar.set_description("Processing %s" % feature)
    # print feature
    # data(feature)
    not_string(feature)
    is_null(feature)
    if length(feature) == False:
        for item in df[feature].unique():

            if len(item) == 1:
                df[feature] = df[feature].replace(item, "0" + item)

def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['F', 'M'], [1, 0])
            
        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

# We have enough data points to work with

# Correlation matrix




# Separate the data into feature data and target data (X_all and y_all, respectively)
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

# One Hot Enoncode Categorial Data using Numpies:
for item in tqdm(y_all.unique()):
    y_all = y_all.replace(2, 0)
X_all = preprocess_features(X_all)

print("Data shape: ", X_all.shape)
print X_all.head()
#Success!!
import matplotlib.pyplot as plt

df = X_all.join(y_all)
# def plot_corr(df, size=10):
#     '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

#     Input:
#         df: pandas DataFrame
#         size: vertical and horizontal size of the plot'''

#     corr = df.corr()
#     fig, ax = plt.subplots(figsize=(size, size))
#     ax.matshow(corr)
#     plt.xticks(range(len(corr.columns)), corr.columns)
#     plt.yticks(range(len(corr.columns)), corr.columns)
#     plt.show()
# plot_corr(df)
df_corr = df.corr().abs()
# plt.matshow(df_corr)

print df_corr

s = df_corr.unstack()
so = s.sort(kind="quicksort")
df__corr = so[-4470:-4460]
df_corr[['C_SEV']].plot(kind='bar')
plt.show()
raw_input("Press Enter to continue...")
# Progress bar for progress visualization
bar = progressbar.ProgressBar(maxval=20, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
#PCA
def doPCA(x):
    
    from sklearn import linear_model, decomposition, datasets
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    bar = progressbar.ProgressBar(maxval=20, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    logistic = linear_model.LogisticRegression()

    pca = decomposition.PCA()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

    digits = datasets.load_digits()
    pca.fit(X_all)

    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')

    n_components = [100, 150, 200, 250, 300]
    Cs = np.logspace(-4, 4, 3)


    estimator = get_params(pipe, dict(pca__n_components=n_components, logistic__C=Cs))

    print("Estimating number of components...")
    bar.start()
    estimator.fit(X_all, y_all)
    bar.end()

    plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='n_components chosen')
    plt.legend(prop=dict(size=12))
    plt.show()
    ncomp = estimator.best_estimator_.named_steps['pca'].n_components
    pca = PCA(n_components=ncomp)
    X_all = pca.fit_transform(X_all)




# Cross Validation:
from sklearn.model_selection import train_test_split
# Scoring Metrics
from sklearn.metrics import f1_score


X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, stratify=y_all, 
                                                    test_size=0.24, random_state=42)
print "\n"
print "Train set 'fatal' pct = {:.2f}%".format(100 * (y_train == 1).mean())
print "Test  set 'fatal' pct = {:.2f}%".format(100 * (y_test == 1).mean())
# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])



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
    bar.start()
    start = time()
    y_pred = clf.predict(features)
    end = time()
    bar.finish()
    
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
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# TODO: Initialize the three models
clf_A = GaussianNB()
clf_B = DecisionTreeClassifier()
clf_C = LogisticRegression(random_state = 30)
clf_D = SGDClassifier(shuffle=True, learning_rate="optimal", penalty='l2', random_state=42)


# TODO: Execute the 'train_predict' function for each classifier and each training set size
# train_predict(clf, X_train, y_train, X_test, y_test)
for clf in [clf_A, clf_B, clf_C, clf_D]:
# for clf in [clf_B]:
    print "\n{}: \n".format(clf.__class__.__name__)
    train_predict(clf, X_train, y_train, X_test, y_test)

# So it woked. Not bad results:
# GaussianNB:           f1 = 0.6097
# DecisionTree:         f1 = 0.7711
# LogisticRegression:   f1 = 0.7780
# SGDClassifier:        f1 = 0.7795


# TODO: Import 'GridSearchCV' and 'make_scorer'
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
# from sklearn.estimator import get_params



parameters = {
    'C': np.logspace(0,4,10),
    'penalty':['l1', 'l2'],
    'class_weight':[None, 'balanced'],
    }
scv = StratifiedShuffleSplit(y_train, test_size=0.25)

# Initialize the classifier
clf = LogisticRegression(random_state=30)

# Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score, pos_label=1)

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
# grid_obj = GridSearchCV(clf, parameters, cv=scv, scoring=f1_scorer)

grid_obj = GridSearchCV(clf, parameters, cv=scv,
                        scoring=f1_scorer, verbose=1,
                        n_jobs=-1, pre_dispatch='2*n_jobs')
# grid_obj = get_params(clf, parameters, cv=scv,
#                         scoring=f1_scorer, verbose=1,
#                         n_jobs=-1, pre_dispatch='2*n_jobs')

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))