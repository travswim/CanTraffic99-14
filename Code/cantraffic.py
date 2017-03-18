""" Capstone Canadian Traffic Data 1999 - 2014:

This project aims to determine what factors lead to fatalities in traffic
accidents using machine learning techniques lerned in the Udacity Machine
Learning Nanodegree program. The objective of cantraffic.py is plot and graph
various statistics of the data available in NCDB_1999_to_2014.csv """

# Import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timing
# import os.path
import pickle


# if os.path.isfile('cantraffic.pk1') is False:
#     df = pd.read_csv("NCDB_1999_to_2014.csv")
#     df.to_pickle('cantraffic.pk1')
# else:
#     df = pd.read_pickle('cantraffic.pk1')
# Read in .csv file
df = pd.read_csv("NCDB_1999_to_2014.csv")
fat = []
nonfatal = []
def fatal(x, year):
    """ fatal(int, int)

        x = 1, fatal
            2, non fatal
        year: 199-2014 """

    return(df['C_YEAR'] == year).loc[(df['C_SEV'] == x)].sum()

male = []
female = []
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

# def plotcharts(start_year, end_year, stat):
#
#     N = end_year - start_year;
#
#     menMeans = (20, 35, 30, 35, 27)
#     womenMeans = (25, 32, 34, 20, 25)
#     # menStd = (2, 3, 4, 1, 2)
#     # womenStd = (3, 5, 2, 3, 3)
#     ind = np.arange(N)    # the x locations for the groups
#     width = 0.35       # the width of the bars: can also be len(x) sequence
#
#     p1 = plt.bar(ind, menMeans, width, color='#d62728', yerr=menStd)
#     p2 = plt.bar(ind, womenMeans, width, bottom=menMeans, yerr=womenStd)
#
#     plt.ylabel('Scores')
#     plt.title('Scores by group and gender')
#     plt.legend((p1[0], p2[0]), ('Men', 'Women'))
#     plt.xticks(ind, tuple(map(str, range(start_year, end_year + 1))))
#     plt.yticks(np.arange(0, 5000, 200))
#
#
#     plt.show()

for year in range(start_year, end_year + 1):
    fat.append(fatal(1, year))
    nonfat.append(fatal(2, year))
print fat
