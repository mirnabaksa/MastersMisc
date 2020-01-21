import time
import math
from os import listdir
from os.path import isfile, join

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE

import matplotlib.patheffects as PathEffects
import seaborn as sns
import numpy as np



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


def showPlot(train, validation = None, filename = "figures/loss.png"):
    plt.plot(train)
    if validation:
        plt.plot(validation)

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')

    plt.savefig(filename)
    plt.close()


import csv
def constructDatasetCSV(root_dir):
    with open('csv/dataset.csv', 'w') as dataset_file:
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

import random       
import collections
def constructTripletDatasetCSV(root_dir):
    with open('csv/dataset_triplet.csv', 'w') as dataset_file:
        file_writer = csv.writer(dataset_file)

        data = collections.defaultdict(list)
        for sub_dir in listdir(root_dir):
            label = sub_dir.replace("_reference_DeepSimu", "")
        
            count = 0
            target_dir = join(root_dir, sub_dir, "fast5")
            for filename in listdir(target_dir):
                data[label].append(join(target_dir, filename))
        
        for label, files in data.items():
            for i in range(100):
                anchor = random.choice(files)
                positive = random.choice(files)

                for negative_label, negative_files in data.items():
                    if label == negative_label:
                        continue
                    negative = random.choice(negative_files)
                    file_writer.writerow((anchor, positive, negative, label))

        


def knn(X, y,  k = 3):
    print("Fitting KNN...")
    neigh = KNeighborsClassifier(n_neighbors = k)
    neigh.fit(X, y)
    return neigh


def visualize(X, y, distinct_labels):
    X = np.array(X)
    y = np.array(y)
    print("Visualizing...")
    tsne = TSNE()
    train_tsne_embeds = tsne.fit_transform(X)
    scatter(train_tsne_embeds, y, distinct_labels, "Data")


def scatter(x, labels, distinct_labels, subtitle=None):
    print("Scattering...")
    palette = np.array(sns.color_palette("hls", 10))
    colors = []
    for label in labels:
        colors.append(palette[distinct_labels.index(label)])

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=colors)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    ax.axis('off')
    ax.axis('tight')
    
    txts = []
    for label in distinct_labels:
        xtext, ytext = np.median(x[labels == label, :], axis=0)
        txt = ax.text(xtext, ytext, label, fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
        
    if subtitle != None:
        plt.suptitle(subtitle)
        
    plt.savefig("figures/tsne.png")

