import torch
from torch.utils.data import Dataset, DataLoader

import h5py
import pandas as pd
import os
import numpy as np
from math import sin, cos

import random
from random import uniform, randint, choice

from sklearn.preprocessing import minmax_scale
from gensim.summarization.summarizer import summarize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 10.0
EOS_token = 10.0

class TripletTestDataset(Dataset):
    def __init__(self):
        n_classes = 3
        n_points = 100
        x = uniform(-0.5,0.5)

        self.positive = []
        self.negative = []
        for i in range(n_points):
            test_sample = []
            x = uniform(-0.5,0.5)
            #x_len = 4
            x_len = randint(40, 70)
            #self.data.append(([min(max(x+0.1*i, -1),1) for i in range(x_len)], 1))
            
            self.positive.append(([sin(0.1* x+0.05*i) for i in range(x_len)], "positive"))

        for i in range(n_points):
            test_sample = []
            x = uniform(-0.5,0.5)
            #x_len = 4
            x_len = randint(40, 70)
            #self.data.append(([min(max(x-0.2*i, -1), 1) for i in range(x_len)], 2))
            self.negative.append(([cos(x-0.08*i) for i in range(x_len)], "negative"))

        self.neutral = []
        for i in range(n_points):
            test_sample = []
            x = uniform(-0.5,0.5)
            #x_len = 4
            x_len = randint(40, 70)
            #self.data.append(([min(max(x-0.2*i, -1), 1) for i in range(x_len)], 2))
            self.neutral.append(([sin(2*x+0.1*i) for i in range(x_len)], "neutral"))


        self.triplets = []
        for i in range(n_points//2):
            a = self.positive[randint(0,99)]
            p = self.positive[randint(0,99)]
            n = self.negative[randint(0,99)]
            self.triplets.append((a,p,n, 1))

        for i in range(n_points//2):
            a = self.positive[randint(0,99)]
            p = self.positive[randint(0,99)]
            n = self.neutral[randint(0,99)]
            self.triplets.append((a,p,n, 1))

        for i in range(n_points//2):
            a = self.negative[randint(0,99)]
            p = self.negative[randint(0,99)]
            n = self.positive[randint(0,99)]
            self.triplets.append((a,p,n, 2))

        for i in range(n_points//2):
            a = self.negative[randint(0,99)]
            p = self.negative[randint(0,99)]
            n = self.neutral[randint(0,99)]
            self.triplets.append((a,p,n, 2))

        for i in range(n_points//2):
            a = self.neutral[randint(0,99)]
            p = self.neutral[randint(0,99)]
            n = self.negative[randint(0,99)]
            self.triplets.append((a,p,n, 3))

        for i in range(n_points//2):
            a = self.neutral[randint(0,99)]
            p = self.neutral[randint(0,99)]
            n = self.positive[randint(0,99)]
            self.triplets.append((a,p,n, 3))

        ''' 
        np.set_printoptions(precision=5)      
        print(np.array(self.data[1][0]))
        print(np.array(self.data[151][0]))
        print(np.array(self.data[201][0]))
        print(np.array(self.data[354][0]))
        '''

        self.n = n_points * 3

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        a, p, n, l = self.triplets[idx]
        return torch.cuda.FloatTensor([[i] for i in a[0]]), torch.cuda.FloatTensor([[i] for i in p[0]]), torch.cuda.FloatTensor([[i] for i in n[0]]), l

    def get_distinct_labels(self):
        return [1,2, 3]
        

class TestDataset(Dataset):

    def __init__(self):
        n_classes = 4
        n_points = 50

        self.data = []
        for i in range(n_points):
            test_sample = []
            x = uniform(-0.2,0.2)
            x_len = randint(7,15)
            #self.data.append(([min(max(x+0.1*i, -1),1) for i in range(x_len)], 1))
            #self.data.append(([0.2 for i in range(x_len)], 1))
            self.data.append(([sin(0.1* x+0.05*i) for i in range(x_len)],1))

        for i in range(n_points):
            test_sample = []
            x = uniform(-0.2,0.2)
            x_len = randint(7,15)
            #self.data.append(([min(max(x-0.2*i, -1), 1) for i in range(x_len)], 2))
            #self.data.append(([0.7 for i in range(x_len)], 2))
            self.data.append(([cos(x-0.08*i) for i in range(x_len)],2))    
     
        for i in range(n_points):
            test_sample = []
            x = uniform(-0.2,0.2)
            x_len = randint(7,15)
            #self.data.append(([min(max(2*x - 0.5*i, -1), 1) for i in range(x_len)], 3))
            #self.data.append(([-0.3 for i in range(x_len)], 3))
            self.data.append(([sin(2*x+0.1*i) for i in range(x_len)],3))   

        for i in range(n_points):
            test_sample = []
            x = uniform(-0.2,0.2)
            x_len = randint(7,15)
            #self.data.append(([min(max(-x + 0.3*i, -1), 1) for i in range(x_len)], 4))
            #self.data.append(([-0.5 for i in range(x_len)], 4))
            self.data.append(([cos(-2*x+0.25*i) for i in range(x_len)],4))   

        ''' 
        np.set_printoptions(precision=5)      
        print(np.array(self.data[1][0]))
        print(np.array(self.data[151][0]))
        print(np.array(self.data[201][0]))
        print(np.array(self.data[354][0]))
        '''

        self.n = n_classes * n_points

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        data, label = self.data[idx]
        #data = minmax_scale(data)
        copy = data.copy()
        copy.insert(0, SOS_token)
        return torch.FloatTensor([[i] for i in copy]), label

    def get_distinct_labels(self):
        return [1,2,3,4]

class SignalTripletDataset(Dataset):

    def __init__(self, root_dir, reference_csv, raw = True):
        self.root_dir = root_dir
        self.reference = pd.read_csv(reference_csv, delimiter = ",", header = None)
        self.raw = raw
        self.n = len(self.reference)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        anchor, positive, negative, label = self.reference.iloc[idx]
        a, p, n = Signal(anchor), Signal(positive), Signal(negative)
        data_a = a.get_raw() if self.raw else a.get_pA()
        data_p = p.get_raw() if self.raw else p.get_pA()
        data_n = n.get_raw() if self.raw else n.get_pA()
        return torch.cuda.FloatTensor([[i] for i in data_a]), torch.cuda.FloatTensor([[i] for i in data_p]), torch.cuda.FloatTensor([[i] for i in data_n]), label

    def get_distinct_labels(self):
        return ["bacillus_anthracis", "ecoli", "klebsiella_pneumoniae", "pantonea_agglomerans", "pseudomonas_koreensis", "yersinia_pestis"]

class SignalDataset(Dataset):

    def __init__(self, root_dir, reference_csv, raw = True):
        self.root_dir = root_dir
        self.reference = pd.read_csv(reference_csv, delimiter = ",", header = None)
        self.raw = raw
        self.n = len(self.reference)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        read_name, label = self.reference.iloc[idx]
        signal = Signal(read_name)
        data = signal.get_raw() if self.raw else signal.get_pA()
        return torch.FloatTensor([[i] for i in data]), label

    def get_distinct_labels(self):
        return ["bacillus_anthracis", "ecoli", "klebsiella_pneumoniae", "pantonea_agglomerans", "pseudomonas_koreensis", "yersinia_pestis"]


class Signal():
    def __init__(self, filename):
        f = h5py.File(filename, 'r')
        self.raw = np.array(f["Raw"]["Reads"]["Read_981"]["Signal"]).astype(np.float)
        self.metadata = f["UniqueGlobalKey"]["channel_id"]
        self.offset = self.metadata.attrs['offset']

        range = self.metadata.attrs['range']
        quantisation = self.metadata.attrs['digitisation']
        self.scale = range/quantisation
    
    def get_raw(self):
        return minmax_scale(self.raw[:400])
        '''
        y = uniform(-1,1)
        flag = choice([0, 1])
        x = None
        x = [y, y+0.1, y+0.2, y+0.3, y+0.4] if flag == 0 else [y, y-0.1, y-0.2, y-0.3, y-0.4] 
        return x[:randint(2,5)], flag
        '''

    def get_pA(self):
        return minmax_scale([self.scale * (raw + self.offset) for raw in self.raw])