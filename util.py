import time
import math
from os import listdir
from os.path import isfile, join

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def showPlot(train, validation = None, filename = "loss.png"):
    plt.plot(train)
    plt.plot(validation)

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')

    plt.savefig(filename)
    plt.close()


import csv
def constructDatasetCSV(root_dir):
    with open('dataset.csv', 'w') as dataset_file:
        file_writer = csv.writer(dataset_file)
        for sub_dir in listdir(root_dir):
            label = sub_dir.replace("_reference_DeepSimu", "")
        
            count = 0
            target_dir = join(root_dir, sub_dir, "fast5/")
            for filename in listdir(target_dir):
                count += 1
                file_writer.writerow((join(target_dir, filename), label))
                if count == 100:
                    break

            
   # f = [join(root_dir,f) for f in listdir(root_dir) if isfile(join(root_dir, f))]
    #np.savetxt("test.csv", f, delimiter="\n", fmt='%s')