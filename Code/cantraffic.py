""" Capstone Canadian Traffic Data 1999 - 2014:

This project aims to determine what factors lead to fatalities in traffic
accidents using machine learning techniques lerned in the Udacity Machine
Learning Nanodegree program. The objective of cantraffic.py is plot and graph
various statistics of the data available in NCDB_1999_to_2014.csv """

# Import statements
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.patches as mpatches
import timing
import math
from itertools import product
# import os.path
import pickle
import json
from functools import partial
# if os.path.isfile('cantraffic.pk1') is False:
#     df = pd.read_csv("NCDB_1999_to_2014.csv")
#     df.to_pickle('cantraffic.pk1')
# else:
#     df = pd.read_pickle('cantraffic.pk1')
# Read in .csv file
df = pd.read_csv("NCDB_1999_to_2014.csv")
# feature = 'C_WDAY'
# r = 0
# for item in df[feature]:
#     if type(item) is not str:
#         df.set_value(r, feature, str(item))
#     else: continue
#     r+=1

start = 1999
end = 2014
fatals = (df['C_SEV'] == 1).sum()
nonFatal = (df['C_SEV'] == 2).sum()
total_accidents = len(df['C_SEV'])
def fatal(x, year):
    """ fatal(int, int)

        x = 1, fatal
            2, non fatal
        year: 1999-2014 """

    return(df['C_YEAR'] == year).loc[(df['C_SEV'] == x)].sum()

def sex(x, sex, year):
    """ sex(int, string, int)
        sex =   'F', female
                'M', male """

    return((df['C_YEAR'] == year)&(df['P_SEX'] == sex)).loc[(df['C_SEV'] == x)].sum()

def weather(x, w, year):
    """ weather(int, string, int)

        w = 1, clear and sunny
            2, overcast, cloudy but no precipitation
            3, raining
            4, snowing, no drifting snow
            5, freezing rain, sleet, hail
            6, limited visibility (eg. drifting snow, fog, smog, dust, smoke)
            7, strong wind
            Q, other
            U, unknown
            X, No data """

    return((df['C_YEAR'] == year)&(df['C_WTHR'] == w)).loc[(df['C_SEV'] == x)].sum()

def vtype(x, v, year):
    """ vtype(int, string, int)

        v = 01, Ligth duty vehicle
            05, Panel/cargo van
            06, other tucks/vans <= 4536Kg GVWR
            07, Unit truck > 4536Kg GVWR
            08, Road tracktor
            09, School Bus
            10, Small school Bus
            11, Bus
            14, Motorcycle/Moped
            16, Off road vehicle
            17, Bicycle
            18, Purpose built motorhome
            19, Farm equipment
            20, Construction equipment
            21, Fire engine
            22, Snowmobile
            23, Street car
            NN, Not applicable
            QQ, other
            UU, unknown
            XX, Not available"""

    return((df['C_YEAR'] == year)&(df['V_TYPE'] == v)).loc[(df['C_SEV'] == x)].sum()

def night(x, h, year):
    """ night(int, string, int)
        h (hour) = 00, 12am
            01, 1am
            .
            .
            .
            23, 11pm
            UU, unknown
            XX, Not available """

    return((df['C_YEAR'] == year)&(df['C_HOUR'] == h)).loc[(df['C_SEV'] == x)].sum()

def month(x, m, year):
    """ month(int, string, int)

        m = 01, Jan
            ...
            12, Dec
            UU, unknown
            XX, Not available"""

    return((df['C_YEAR'] == year)&(df['C_MNTH'] == m)).loc[(df['C_SEV'] == x)].sum()

def day(x, d, year):
    """ day(int, string, int)

        d = 1, Mon
        ...
            7, Sunday
            U, unknown
            X, Not available"""

    return((df['C_YEAR'] == year)&(df['C_WDAY'] == d)).loc[(df['C_SEV'] == x)].sum()

def pop_random(lst):
    """ Pop a random item for a given list"""
    idx = random.randrange(0, len(lst) +1 )
    return lst.pop(idx)

def rounder(num):
    """ This function is used to find the y-axis limit on the stacked
    bar plot """
    a = list(map(int,str(num)))
    x = math.pow(10, len(a)-1)
    return int((a[0]+1)*x)

def plotfatal():
    """ Stacked bar chart of fatal and Non-Fatal accidents in Canada"""
    fat = []
    nonfat = []
    max_feature = []
    year = 'C_YEAR'
    # For every year, calculate the number of fatalities/non-fatalities
    for item in df[year].unique():
        fat.append(fatal(1, item))
        nonfat.append(fatal(2, item))
        # Calculate the maximum value of fatal + Non-Fatal for any given year
    max_feature = [x + y for x, y in zip(fat, nonfat)]
    # Calculate percentage of fatalities
    percentage = map(float, np.array(fat))/(np.array(fat) + np.array(nonfat))*100

    print [round(elem, 2) for elem in percentage]

    N = end - start + 1;    # Year range
    ind = np.arange(N)      # the x locations for the groups
    width = 0.50       # the width of the bars: can also be len(x) sequence
    # Plotting stuff
    p1 = plt.bar(ind, tuple(fat), width, color='r')
    p2 = plt.bar(ind, tuple(nonfat), width, color = 'y', bottom = fat)
    plt.ylabel('Number of accidents')
    plt.title('Canadian Fatal Vehicle Accidents by Year')
    plt.xticks(ind, tuple(map(str, range(start, end + 1))))
    plt.yticks(np.arange(0, rounder(max(max_feature)), rounder(max(max_feature))/10))
    plt.legend((p1, p2), ('Fatal', 'Non-Fatal'))
    plt.show()

def plotfatalalities():
    fat = []
    max_feature = []
    year = 'C_YEAR'
    for item in df[year].unique():
        fat.append(fatal(1, item))
    N = end - start + 1;    # Year range
    ind = np.arange(N)      # the x locations for the groups
    width = 0.50       # the width of the bars: can also be len(x) sequence
    # Plotting stuff
    p1 = plt.bar(ind, tuple(fat), width, color='r')
    plt.ylabel('Number of fatalities')
    plt.title('Canadian Fatal Vehicle Accidents by Year')
    plt.xticks(ind, tuple(map(str, range(start, end + 1))))
    plt.yticks(np.arange(0, rounder(max(fat)), rounder(max(fat))/10))
    # plt.legend((p1, ), ('Fatal', 'Non-Fatal'))
    plt.show()

def plot(feature):
    # plt.ylabel('Number of fatalities')
    # plt.title('Canadian Fatal Vehicle Accidents by Year')
    # plt.xticks(ind, tuple(map(str, range(start, end + 1))))
    # plt.yticks(np.arange(0, rounder(max(max_feature)), rounder(max(max_feature))/10))
    # plt.legend((p1, p2), ('Fatal', 'Non-Fatal'))
    # plt.show()
    # Setup Variables, Lists, & Dictionaries for plots
    N = end - start + 1 # Year range
    ind = np.arange(N)
    width = 0.5
    weatherItem = {}
    pairs = {}
    percentage = {}
    p = {}
    colours =[]
    label = []
    val = []
    items = []
    bottoms = []
    max_feature = [0] * N

    yaxis = {'C_WTHR': 'Weather', 'V_TYPE': 'Vehicle Type', 'P_SEX': 'Sex', \
    'C_HOUR': 'Hour of the Day', 'C_MNTH': 'Month of the Year', \
    'C_WDAY': 'Day of the Week'}

    # Color randomiation for plots
    for name, hex in mcolors.cnames.iteritems():
        colours.append(name)
    random.shuffle(colours)

    # Read .json file/legend
    with open('weather.json') as data:
        dt = json.load(data)
    its = dt['ITEMS'][feature]

    # Match item variable with a color for plot
    for item in its:
        pairs["{0}".format(item)] = colours.pop()

    #   Cleanup dataframe entries that are of the wrong type or are not properly
    # labeled. Ex. '1' --> '01'
    df1 = df[feature].apply(str)
    if feature == 'C_MNTH':
        for item in df1.unique():
            if len(item) < 2:
                    df1 = df1.replace([item], '0' + item)
            else: continue
    # Create list of fatalities for a given year & item
    for item in df1.unique():
        bottoms.append(item)    # List of items for legend
        lst = []                # Fatalities list per item, per year

        for year in range(start, end + 1):
            # Legend for the above function calculations
            legend = {'C_WTHR': partial(weather, 1, item, year),\
                    'V_TYPE': partial(vtype, 1, item, year), \
                    'P_SEX': partial(sex, 1, item, year),\
                    'C_HOUR': partial(night, 1, item, year), \
                    'C_MNTH': partial(month, 1, item, year), \
                    'C_WDAY': partial(day, 1, item, year)}
            lst.append(legend[feature]())

        # Store in dictionary based on item
        weatherItem["{0}".format(item)] = lst
        label.append(its[item])

        # Create plot variables 'p':
        p["p{0}".format(item)] = plt.bar(ind, tuple(weatherItem[item]), width, color = pairs[item], \
            bottom = tuple(max_feature))

        # Keep track of the total fatalities for a given year
        max_feature = [x + y for x, y in zip(max_feature, list(weatherItem[item]))]
        # Store plot variables for later
        val.append(p["p{0}".format(item)])

    # Create new DataFrame for more visualization/comparison
    newDF = pd.DataFrame(weatherItem)
    years = range(start, end + 1)
    headers = list(newDF.columns.values)

    # Rename column headers with names instead of numbers
    for key, value in its.items():
        for header in headers:
            if key == header:
                newDF.rename(columns={key:value}, inplace=True)
            else: continue

    newDF['Total'] = newDF.sum(axis=1)
    newDF['Year'] = years
    print newDF

    # Plot labels
    plt.ylabel('Number of accidents')
    plt.xlabel('Year')
    plt.title("Canadian Fatal Vehicle Accidents by Year: {0}".format(yaxis[feature]))
    plt.xticks(ind, tuple(map(str, range(start, end + 1))))
    plt.yticks(np.arange(0, rounder(max(newDF['Total'])), rounder(max(newDF['Total']))/10))
    plt.legend(tuple(val), tuple(label), loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
# Call plot function
# plot('P_SEX')
plot('C_WTHR')
plot('V_TYPE')

plotfatal()
plotfatalalities()
print("Total number of Accidents: " + str(total_accidents))
print("Total fatalities: " + str(fatals))
print("Total non-fatalities: " + str(nonFatal))
