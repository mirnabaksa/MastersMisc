import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from Model import TripletEncoder
from SignalDataset import SignalTripletDataset

from util import knn, visualize, constructTripletDatasetCSV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed(1)

def collate(input):
    a, p, n, labels = map(list, zip(*input))
    a = get_input(a)
    p = get_input(p)
    n = get_input(n)
    
    return a, p, n, labels

def get_input(in_batch):
    padded = pad_sequence(in_batch, batch_first = True, padding_value = 0)
    lens = [len(x) for x in in_batch]
    out = pack_padded_sequence(padded, lens, batch_first = True, enforce_sorted = False)
    return out

def triplet_loss(a, p, n, margin=0.2) : 
    d = nn.PairwiseDistance(p=2)
    distance = d(a, p) - d(a, n) + margin 
    loss = torch.mean(torch.max(distance, torch.zeros_like(distance))) 
    return loss


def train(train_dataset, validation_dataset = None, iterations = 10, hidden_size = 64, batch_size = 64):
    print("Training...")
    train = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate)
    if validation_dataset:
        validation = DataLoader(validation_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate)

    encoder = TripletEncoder(1, hidden_size).to(device)
    optimizer = optim.Adam(encoder.parameters())
    criterion = triplet_loss

    train_losses = []

    for iter in range(iterations):
        loss_acc = 0
        enc_last_hidden = None
        
        for in_a, in_p, in_n, l in train:  
            optimizer.zero_grad()

            a, p, n = encoder(in_a, in_p, in_n, None)
            enc_last_hidden = a
            
            loss = criterion(a, p, n)
            loss_acc += loss.item()
      
            loss.backward(retain_graph = True)
            optimizer.step()
        
        train_losses.append(loss_acc/len(train))


        if iter%1 == 0:
            print("Iteration:", iter, 
            " Train loss: ", "{0:.5f}".format(loss_acc/len(train))
            )
        loss_acc = 0
        
    torch.save(encoder, "triplet_encoder.pt")

def evaluate(test_dataloader):
    model = torch.load("model.pt")
    for a,p,n,l in test_dataloader:
        output = model.get_latent(a)
        print("Target")
        print(a.squeeze())
        print("Model out")
        print(output.squeeze())
        print()

def get_latent(dataloader, model):
    print("Collecting latent vector...")
    X = []
    y = []
    for a,p,n,label in dataloader:
        hidden = model.get_latent(a)
        vector = hidden.squeeze().tolist()
        if dataloader.batch_size == 1:
            vector = [vector]
        
        X.extend(vector)
        y.extend(label) 
    
    return X, y


def predict(predictor, model, dataloader):
    print("Predicting...")
    correct = 0
    for a, p, n, l in dataloader:
        X = a
        labels = l
 
        encoder_hidden = model.get_latent(X)

        X = encoder_hidden.squeeze().tolist()
        if dataloader.batch_size == 1:
            X = [X]
        pred = predictor.predict(X)
        correct += sum(pred == labels.item())
    
    n = len(dataloader) * dataloader.batch_size
    print("Accuracy: ", correct / n)

if __name__ == '__main__':
    #constructTripletDatasetCSV("../Signals/full_dataset/")
    
    dataset = SignalTripletDataset("../Signals/full_dataset/", "dataset_triplet.csv", raw = True)
    train_size = int(0.8 * len(dataset))
    val_test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_test_size])  
    train(train_dataset)

    dataloader = DataLoader(train_dataset, batch_size = 1, shuffle = True, collate_fn = collate)
    model = torch.load("triplet_encoder.pt")

    X, y = get_latent(dataloader, model)
    predictor = knn(X, y, 3)
    visualize(X, y, dataset.get_distinct_labels())

    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = True, collate_fn = collate)
    #evaluate(test_dataloader)
    predict(predictor, model, test_dataloader)
    