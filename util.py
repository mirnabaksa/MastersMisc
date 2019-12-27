import time
import math
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

def timeNow():
    return time.time()

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

import numpy as np
def constructDatasetCSV(root_dir):
    f = [join(root_dir,f) for f in listdir(root_dir) if isfile(join(root_dir, f))]
    np.savetxt("test.csv", f, delimiter="\n", fmt='%s')