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
del fatal, nonFatal, total_accidents
#Find the feature columns
feature_cols = list(df.drop('C_SEV', 1))
#Find the target columns
target_col = df['C_SEV']
#Print some more stuff
# print "Feature columns:\n{}".format(feature_cols)
# print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
try:
    X_all = df[feature_cols]
    y_all = df['C_SEV']
    # print "Target: {}".format(y_all)
    # print "Features: {}".format(feature_cols)

    print "Successfully separated features and target"
    del target_col
except:
    print "Failed to separate features and target"


# Show the feature information by printing the first five rows
# print "\nFeature values:"
# print X_all.head()

# This function is to replace wrong input data with the correct ones
# (it would have been easier if the data had been entered correctly :/)
def replace(item, newitem):
    df[feature].replace(item, newitem)
# Preprocessing: Lets see if the data is uniform
# unique_items = {}
# for feature in feature_cols:
#     for item in list(df[feature].unique()):
#         if len(item <=2):
#
#     print feature
#     print list(df[feature].unique())
# print list(df['C_SEV'].unique())
# unique = pd.DataFrame(unique_items)
# print unique
for feature in feature_cols:
    # print feature
    lst = list(df[feature].unique())
    for item in lst:
        if re.match('[1-9]', item) != None:
            repla = "0" + item
            ## need to use regex to match numbers, otherwise might not work for UU, U or XX, X
            ## also need to deal with NaN or N/A or blank spaces
            print item print repla
            lst.remove(item)
                df[feature].replace(item, repla)
# df[year].unique():
# def data_fix()
