import torch
from torch.utils.data import Dataset, DataLoader

import h5py
import pandas as pd
import os
import numpy as np

class SignalDataset(Dataset):

    def __init__(self, root_dir, reference_csv, raw = True):
        self.root_dir = root_dir
        self.reference = pd.read_csv(reference_csv)
        self.raw = raw
        self.n = sum([len(files) for r, d, files in os.walk(self.root_dir)])

    def __len__(self):
        return 10#self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        read_name = self.reference.iloc[idx,0]
        signal = Signal(read_name)
        data, label = signal.get_raw() if self.raw else signal.get_pA()
        return torch.FloatTensor([[i] for i in data]), label


from random import uniform, randint, choice
from sklearn.preprocessing import minmax_scale
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
        #print(self.raw[:20])
        #print(minmax_scale(self.raw[:20]))
        #exit(0)
        #return 
        #return minmax_scale(self.raw[:150])
        
        y = uniform(-1,1)
        flag = choice([0, 1])
        x = None
        x = [y, y+0.1, y+0.2, y+0.3, y+0.4] if flag == 0 else [y, y-0.1, y-0.2, y-0.3, y-0.4] 
        return x[:randint(2,5)], flag


    def get_pA(self):
        return [self.scale * (raw + self.offset) for raw in self.raw]