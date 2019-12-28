import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torch.autograd import Variable

from util import asMinutes, timeSince, showPlot, timeNow, constructDatasetCSV
from Model import Encoder, Decoder
from SignalDataset import Signal, SignalDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate(input):
    in_batch, labels = map(list, zip(*input))
   
    padded = pad_sequence(in_batch, batch_first = True, padding_value = 0)
    lens = [len(x) for x in in_batch]
    out = pack_padded_sequence(padded, lens, batch_first = True, enforce_sorted = False)
    return out, padded.squeeze(), labels
  

def train(dataset, iterations = 10, hidden_size = 5, batch_size = 5):
    dataloader = DataLoader(dataset, batch_size = batch_size, collate_fn = collate)
    encoder = Encoder(1, hidden_size).to(device)
    decoder = Decoder(hidden_size, 1).to(device)

    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())
    criterion = nn.MSELoss()
    iters_per_epoch = len(dataset)/batch_size

    for iter in range(iterations):
        loss_acc = 0
        last_hidden = None
        
        for input, target, _ in dataloader:  
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(input, last_hidden)
            last_hidden = encoder_hidden
            decoder_outputs = decoder(encoder_outputs, None)
            
            loss = criterion(decoder_outputs, target)
            loss_acc += loss
      
            loss.backward(retain_graph = True)
            encoder_optimizer.step()
            decoder_optimizer.step()
        
        if iter%100 == 0:
            print("Iter :", iter, " Loss: ", loss_acc/iters_per_epoch)
        loss_acc = 0

    torch.save(encoder, "encoder.pt")
    torch.save(decoder, "decoder.pt")


def decode(input,target, encoder, decoder):
    print(target)
    encoder_outputs, encoder_hidden = encoder(input, None)
    decoder_outputs = decoder(encoder_outputs, None)
    print(decoder_outputs)

def evaluate(dataloader):
    encoder = torch.load("encoder.pt")
    decoder = torch.load("decoder.pt")
    encoder.eval()
    decoder.eval()

    y = None
    for batch, target, _ in dataloader:
        y = batch
    decode(y, target, encoder, decoder)

from sklearn.neighbors import KNeighborsClassifier
def knn(encoder, dataloader, k = 3):
    X = []
    y = []
    for input, _, label in dataloader:
        encoder_outputs, encoder_hidden = encoder(input, None)
        vector = encoder_hidden.squeeze().tolist()
        if dataloader.batch_size == 1:
            vector = [vector]
        
        X.extend(vector)
        y.extend(label) 

    neigh = KNeighborsClassifier(n_neighbors= k)
    neigh.fit(X, y)
    return neigh

def predict(predictor, encoder, dataloader):
    correct = 0
    for X_test in dataloader:
        X = X_test[0]
        target = X_test[1]
        labels = X_test[2]
 
        encoder_outputs, encoder_hidden = encoder(X, None)

        X = encoder_hidden.squeeze().tolist()
        if dataloader.batch_size == 1:
            X = [X]
        pred = predictor.predict(X)
        correct += sum(pred == labels)
    
    n = len(dataloader) * dataloader.batch_size
    print("Accuracy: ", correct / n)


if __name__ == '__main__':
    # constructDatasetCSV("../data_ecoli/reads/ecoli/")
    dataset = SignalDataset("../data_ecoli/reads/ecoli/", "test.csv")
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, collate_fn = collate)
    #train(dataset)
    #evaluate(dataloader)
    
    encoder = torch.load("encoder.pt")
    predictor = knn(encoder, dataloader)
    predict(predictor, encoder, dataloader)
    