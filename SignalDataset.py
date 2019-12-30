import torch
from torch.utils.data import Dataset, DataLoader

import h5py
import pandas as pd
import os
import numpy as np

import random
from random import uniform, randint, choice


from sklearn.preprocessing import minmax_scale

class TestDataset(Dataset):

    def __init__(self):
        n_classes = 4
        n_points = 100

        self.data = []
        for i in range(n_points):
            test_sample = []
            x = uniform(-0.5,0.5)
            x_len = randint(4, 7)
            self.data.append(([min(max(x+0.1*i, -1),1) for i in range(x_len)], 1))

        for i in range(n_points):
            test_sample = []
            x = uniform(-0.5,0.5)
            x_len = randint(4, 7)
            self.data.append(([min(max(x-0.2*i, -1), 1) for i in range(x_len)], 2))
    
     
        for i in range(n_points):
            test_sample = []
            x = uniform(-0.5,0.5)
            x_len = randint(4, 7)
            self.data.append(([min(max(2*x - 0.5*i, -1), 1) for i in range(x_len)], 3))

        for i in range(n_points):
            test_sample = []
            x = uniform(-0.5,0.5)
            x_len = randint(4, 7)
            self.data.append(([min(max(-x + 0.1*i, -1), 1) for i in range(x_len)], 4))

        np.set_printoptions(precision=5)      
        print(np.array(self.data[1][0]))
        print(np.array(self.data[151][0]))
        print(np.array(self.data[201][0]))
        print(np.array(self.data[354][0]))
        

        self.n = n_classes * n_points

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        data, label = self.data[idx]
        #data = minmax_scale(data)
        return torch.FloatTensor([[i] for i in data]), label


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
        return minmax_scale(self.raw[:150])
        '''
        y = uniform(-1,1)
        flag = choice([0, 1])
        x = None
        x = [y, y+0.1, y+0.2, y+0.3, y+0.4] if flag == 0 else [y, y-0.1, y-0.2, y-0.3, y-0.4] 
        return x[:randint(2,5)], flag
        '''

    def get_pA(self):
        return minmax_scale([self.scale * (raw + self.offset) for raw in self.raw[:1500]])