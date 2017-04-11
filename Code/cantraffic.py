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
import matplotlib.patches as mpatches
import timing
import math
from itertools import product
# import os.path
import pickle


# if os.path.isfile('cantraffic.pk1') is False:
#     df = pd.read_csv("NCDB_1999_to_2014.csv")
#     df.to_pickle('cantraffic.pk1')
# else:
#     df = pd.read_pickle('cantraffic.pk1')
# Read in .csv file
df = pd.read_csv("NCDB_1999_to_2014.csv")
start = 1999
end = 2014

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

    return((df['C_YEAR'] == year)&(df['C_MNTH'] == h)).loc[(df['C_SEV'] == x)].sum()

def day(x, d, year):
    """ day(int, string, int)

        d = 1, Mon
        ...
            7, Sunday
            U, unknown
            X, Not available"""

    return((df['C_YEAR'] == year)&(df['C_WDAY'] == h)).loc[(df['C_SEV'] == x)].sum()

def pop_random(lst):
    idx = random.randrange(0, len(lst) +1 )
    return lst.pop(idx)

def rounder(num):
    a = list(map(int,str(num)))
    x = math.pow(10, len(a)-1)
    return int((a[0]+1)*x)

def plotfatal():
    fat = []
    nonfat = []
    max_feature = []
    year = 'C_YEAR'
    for item in df[year].unique():
        fat.append(fatal(1, item))
        nonfat.append(fatal(2, item))
    max_feature = [x + y for x, y in zip(fat, nonfat)]
    # for item in df[feature].unique():
    #     print item
    percentage = map(float, np.array(fat))/(np.array(fat) + np.array(nonfat))*100

    print [round(elem, 2) for elem in percentage]

    N = end - start + 1;
    ind = np.arange(N)    # the x locations for the groups
    width = 0.50       # the width of the bars: can also be len(x) sequence
    p1 = plt.bar(ind, tuple(fat), width, color='r')
    p2 = plt.bar(ind, tuple(nonfat), width, color = 'y', bottom = fat)
    plt.ylabel('Number of accidents')
    plt.title('Canadian Fatal Vehicle Accidents by Year')
    plt.xticks(ind, tuple(map(str, range(start, end + 1))))
    plt.yticks(np.arange(0, rounder(max(max_feature)), rounder(max(max_feature))/10))
    plt.legend((p1, p2), ('Fatal', 'Non-Fatal'))
    plt.show()
# plotfatal()


def plot():
    feature = 'C_WTHR'
    N = end - start + 1
    ind = np.arange(N)
    width = 0.5
    weatherItem = {}
    pairs = {}
    percentage = {}
    p = {}
    condition = 'Weather'
    colours = ['r', 'b', 'k', 'w', 'm', 'g', 'c', 'y', 'yellow', 'aqua']
    random.shuffle(colours)
    legend = {'1': 'sunny', '2': 'cloudy', '3': 'rainny', '4': 'snowing', \
    '5': 'sleet', '6': 'visibility', '7': 'windy', 'Q': 'other', 'U': 'unknown',\
    'X': 'N/A'}
    label = []
    val = []
    bottoms = []
    max_feature = [0] * N

    k=0
    for key in legend.keys():
        pairs["{0}".format(key)] = colours.pop()
    for item in df[feature].unique():

        bottoms.append(item)
        # print bottoms[k]
        # print item
        lst = []
        for year in range(start, end + 1):
            lst.append(weather(1, item, year))
        weatherItem["{0}".format(item)] = lst
        # value.append(weatherItem[item])
        label.append(legend[item])
        # print weatherItem[item]
        # print max_feature
        max_feature = [x + y for x, y in zip(max_feature, list(weatherItem[item]))]
        # print max_feature
        if k == 0:

            p["p{0}".format(k)] = plt.bar(ind, tuple(weatherItem[item]), width, color = pairs[item])

        else:
            # print bottoms[x]
            p["p{0}".format(k)] = plt.bar(ind, tuple(weatherItem[item]), width, color = pairs[item], \

            bottom = tuple(weatherItem[bottoms[k-1]]))
            # print bottoms[k]
            # print bottoms[k-1]

        val.append(p["p{0}".format(k)])
        k += 1
        # print k

    newDF = pd.DataFrame(weatherItem)
    years = range(start, end + 1)
    headers = list(newDF.columns.values)

    for key, value in legend.items():
        for header in headers:
            if key == header:
                newDF.rename(columns={key:value}, inplace=True)
            else: continue

    newDF['Total'] = newDF.sum(axis=1)
    newDF['Year'] = years
    # print p
    # print newDF
    plt.ylabel('Number of accidents')
    plt.title("Canadian Fatal Vehicle Accidents by Year due to {0}".format(condition))
    plt.xticks(ind, tuple(map(str, range(start, end + 1))))
    # plt.yticks(np.arange(0, rounder(max(max_feature)), rounder(max(max_feature))/10))
    plt.yticks(np.arange(0, rounder(max(newDF['Total'])), rounder(max(newDF['Total']))/10))
    plt.legend(tuple(val), tuple(label), loc='center left', bbox_to_anchor=(1, 0.5))
    # print tuple(list(value)
    # print tuple(label)

    plt.show()
plot()
# def plotweather(feature, variable, start, end):
#     colours = ['r', 'b', 'k', 'w', 'm', 'g', 'c', 'y', 'yellow', 'aqua']
#     legend = {'1': 'sunny', '2': 'cloudy', '3': 'rainny', '4': 'snowing', \
#     '5': 'sleet', '6': 'visibility', '7': 'windy', 'Q': 'other', 'U': 'unknown',\
#      'X': 'N/A'}
#     N = end - start + 1;
#     pairs = {}
#     l = len(legend)
#     ind = np.arange(N)    # the x locations for the groups
#     width = 0.50
#     # max_feature = [0]*(N)
#     plot = []
#     plot2 = []
#     weatherItem = {}
#     p = {}
#     for key, value in legend.items():
#         pairs["{0}".format(key)] = colours.pop()
#
#     for item in df.feature.unique():
#         lst = []
#         p = []
#         for year in range(start, end + 1):
#             weatherItem["weather_{0}".format(item)] = lst.append(weather(1, item, year))
#             p["p{0}".format(item)].append(ind, tuple("list{0}".format(item)), width, color = pairs[item])
#             plot.append("p{0}[0]".format(item))
#             plot2.append(legend[item])
#         # max_feature = [x + y for x, y in zip(max_feature, "list{0}".format(item)]
#         print "list{0}".format(item)
#
#
#
#            # the width of the bars: can also be len(x) sequence
#     # p1 = plt.bar(ind, tuple(fat), width, color='r')
#     plt.ylabel('Number of accidents')
#     plt.title("Canadian Fatal Vehicle Accidents by Year due to {0}".format(variable))
#     plt.xticks(ind, tuple(map(str, range(start, end + 1))))
#     plt.yticks(np.arange(0, rounder(max(max_feature)), rounder(max(max_feature))/10))
#     plt.legend(tuple(plot), tuple(plot2))
    #  plt.show()
#
