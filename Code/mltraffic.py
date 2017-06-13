""" Machine learning file for Canadian Traffic Data 1999-2014"""
#Import stuff
from sklearn import tree
import numpy as np
import pandas as pd
import re
# import cantraffic as ct

#Read the data file (csv)
df = pd.read_csv("NCDB_1999_to_2014.csv")

print "Data read successfully"

#define the fatalities/non-fatalities
fatal = (df['C_SEV'] == 1).sum()
nonFatal = (df['C_SEV'] == 2).sum()
total_accidents = len(df['C_SEV'])

#Print some stuff
print "Total number of fatalities: {}".format(fatal)

print "Total number of non-fatalities: {}".format(nonFatal)

print "Total number of Accidents: {}".format(total_accidents)
del fatal, nonFatal
#Find the feature columns
feature_cols = list(df.drop('C_SEV', 1))
#Find the target columns
target_col = 'C_SEV'
#Print some more stuff
# print "Feature columns:\n{}".format(feature_cols)
# print "\nTarget column: {}".format(target_col)




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

variable_count = 0

# This section calls various functions above and adds '0' to '1'
for feature in feature_cols:
    # print feature
    # data(feature)
    not_string(feature)
    is_null(feature)
    if length(feature) == False:
        for item in df[feature].unique():

            if len(item) == 1:
                df[feature] = df[feature].replace(item, "0" + item)

# Count the number of unique items to deal with preprocessing, PCA and the curse of dimensionality
    variable_count += len(list(df[feature].unique()))
print "Featire Space: {0}".format(variable_count)
print "Data length: {0}".format(total_accidents)
# We have enough data points to work with

# Separate the data into feature data and target data (X_all and y_all, respectively)
try:
    X_all = df[feature_cols]
    y_all = df['C_SEV']
    # print "Target: {}".format(y_all)
    # print "Features: {}".format(feature_cols)

    print "Successfully separated features and target"
except:
    print "Failed to separate features and target"
