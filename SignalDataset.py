import torch
from torch.utils.data import Dataset, DataLoader

import h5py
import pandas as pd
import os

class SignalDataset(Dataset):

    def __init__(self, root_dir, reference_csv, raw = True):
        self.root_dir = root_dir
        self.reference = pd.read_csv(reference_csv)
        self.raw = raw
        self.n = sum([len(files) for r, d, files in os.walk(self.root_dir)])

    def __len__(self):
        return 100#self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        read_name = self.reference.iloc[idx,0]
        signal = Signal(read_name)
        return torch.FloatTensor([[i] for i in signal.get_raw()]) if self.raw else torch.FloatTensor(signal.get_pA())


from random import uniform, randint
class Signal():
    def __init__(self, filename):
        f = h5py.File(filename, 'r')
        self.raw = f["Raw"]["Reads"]["Read_981"]["Signal"]
        self.metadata = f["UniqueGlobalKey"]["channel_id"]
        self.offset = self.metadata.attrs['offset']

        range = self.metadata.attrs['range']
        quantisation = self.metadata.attrs['digitisation']
        self.scale = range/quantisation
    
    def get_raw(self):
        #return self.raw[:5]
        x = [uniform(0,1), uniform(0,1), uniform(0,1), uniform(0,1), uniform(0,1)]
        return x[:randint(2,5)]

    def get_pA(self):
        return [self.scale * (raw + self.offset) for raw in self.raw]