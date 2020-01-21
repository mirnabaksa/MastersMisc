import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from Model import TripletEncoder
from SignalDataset import SignalTripletDataset, TripletTestDataset

from util import knn, visualize, constructTripletDatasetCSV, showPlot

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


def train(train_dataset, validation_dataset = None, iterations = 60, hidden_size = 128, batch_size = 32):
    print("Training...")
    train = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate)

    if validation_dataset:
        validation = DataLoader(validation_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate)

    encoder = TripletEncoder(1, hidden_size).to(device)
    optimizer = optim.Adam(encoder.parameters())
    criterion = triplet_loss

    train_losses = []
    validation_losses = []

    for iter in range(iterations):
        loss_acc = 0
        enc_last_hidden = None
        encoder.train()
        
        for in_a, in_p, in_n, l in train:  
            a, p, n = encoder(in_a.to(device), in_p.to(device), in_n.to(device), None)
            enc_last_hidden = a
            
            optimizer.zero_grad()
            loss = criterion(a, p, n)
            loss_acc += loss.item()
      
            loss.backward(retain_graph = True)
            optimizer.step()
        
        train_losses.append(loss_acc/len(train))

        
        with torch.no_grad():
            val_loss_acc = 0
            for a,p,n,l in validation:
                input = a.to(device)
            
                encoder.eval()
                a,p,n = encoder(a,p,n)

                val_loss = criterion(a,p,n)
                val_loss_acc += val_loss.item()
            validation_losses.append(val_loss_acc)

        if iter%1 == 0:
            print("Iteration:", iter, 
            " Train loss: ", "{0:.5f}".format(loss_acc/len(train)), 
            " Validation loss: ", "{0:.5f}".format(validation_losses[-1])
            )
        loss_acc = 0
        
    
    showPlot(train_losses, validation_losses)

    torch.save(encoder, "models/triplet_encoder.pt")

def evaluate(test_dataloader):
    model = torch.load("models/triplet_encoder.pt")
    for a,p,n,l in test_dataloader:
        output = model.get_latent(a)
        print("Target")
        print()
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
        print("predicted", pred)
        print("label", labels)
        
        for i in range(len(pred)):
            if pred[i] == labels[i]:
                correct += 1
        #scorrect += sum(pred == labels.item())
    
    n = len(dataloader) * dataloader.batch_size
    print(correct)
    print("Accuracy: ", correct / n)

if __name__ == '__main__':
    #constructTripletDatasetCSV("../Signals/full_dataset/")
    
    dataset = TripletTestDataset()
    #dataset = SignalTripletDataset("../Signals/full_dataset/", "csv/dataset_triplet.csv", raw = True)
    train_size = int(0.8 * len(dataset))
    val_test_size = (len(dataset) - train_size) // 2
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,  val_test_size, val_test_size]) 

    train(train_dataset, validation_dataset)

    dataloader = DataLoader(train_dataset, batch_size = 1, shuffle = True, collate_fn = collate)
    model = torch.load("models/triplet_encoder.pt")

    X, y = get_latent(dataloader, model)
    predictor = knn(X, y, 5)

    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = True, collate_fn = collate)
    #evaluate(test_dataloader)
    predict(predictor, model, test_dataloader)
    
    visualize(X, y, dataset.get_distinct_labels())