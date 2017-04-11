import numpy as np
import matplotlib.pyplot as plt
import operator


N = 5
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
kidMeans = (22, 32, 40, 32, 33)
menStd = (2, 3, 4, 1, 2)
womenStd = (3, 5, 2, 3, 3)
kidStd = (2, 3, 4, 1, 1)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, menMeans, width, color='c', yerr=menStd)
p2 = plt.bar(ind, womenMeans, width, color='r', bottom=menMeans, yerr=womenStd)
p3 = plt.bar(ind, kidMeans, width, color='b', bottom=tuple(map(operator.add, menMeans, womenMeans)), yerr=kidStd)

plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.yticks(np.arange(0, 150, 10))
plt.legend((p1, p2, p3), ('Men', 'Women', 'Kid'))

plt.show()
