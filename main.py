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
from SignalDataset import Signal, SignalDataset, TestDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed(1)

def collate(input):
    in_batch, labels = map(list, zip(*input))
   
    padded = pad_sequence(in_batch, batch_first = True, padding_value = 0)
    lens = [len(x) for x in in_batch]
    out = pack_padded_sequence(padded, lens, batch_first = True, enforce_sorted = False)
    return out, padded.squeeze(), labels
  

def train(train_dataset, validation_dataset = None, iterations = 100, hidden_size = 128, batch_size = 16):
    print("Training...")
    train = DataLoader(train_dataset, batch_size = batch_size, collate_fn = collate, shuffle = True)
    validation = DataLoader(validation_dataset, batch_size = batch_size, collate_fn = collate, shuffle = True)

    encoder = Encoder(1, hidden_size).to(device)
    decoder = Decoder(hidden_size, 1).to(device)

    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())
    criterion = nn.MSELoss()

    train_losses = []
    validation_losses = []

    for iter in range(iterations):
        loss_acc = 0
        enc_last_hidden = None
        dec_last_hidden = None
        
        for input, target, _ in train:  
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, enc_last_hidden = encoder(input, enc_last_hidden)
            decoder_outputs, dec_last_hidden = decoder(encoder_outputs, dec_last_hidden)
            
            loss = criterion(decoder_outputs, target)
            loss_acc += loss.item()
      
            loss.backward(retain_graph = True)
            encoder_optimizer.step()
            decoder_optimizer.step()
        
        train_losses.append(loss_acc/len(train))

    
        with torch.no_grad():
            val_loss_acc = 0
            for input, target, _ in validation:
                input = input.to(device)
                target = target.to(device)
            
                encoder.eval()
                decoder.eval()

                encoder_outputs, enc_last_hidden = encoder(input, enc_last_hidden)
                decoder_outputs, dec_last_hidden = decoder(encoder_outputs, dec_last_hidden)

                val_loss = criterion(decoder_outputs, target)
                val_loss_acc += val_loss.item()
            validation_losses.append(val_loss_acc)
    
        if iter%10 == 0:
            print("Iteration:", iter, 
            " Train loss: ", "{0:.5f}".format(loss_acc/len(train)), 
            " Validation loss: ", "{0:.5f}".format(validation_losses[-1])
            )
        loss_acc = 0
        
    showPlot(train_losses, validation_losses)

    torch.save(encoder, "encoder.pt")
    torch.save(decoder, "decoder.pt")

def evaluate(test_dataloader):
    encoder = torch.load("encoder.pt")
    decoder = torch.load("decoder.pt")
    for input, target, _ in test_dataloader:
        encoder_outputs, encoder_hidden = encoder(input, None)
        decoder_outputs, decoder_hidden = decoder(encoder_outputs, None)
        print("Target")
        print(target)
        print("Model out")
        print(decoder_outputs)
        print()

def decode(encoder, decoder, input, target, encoder_hidden = None, decoder_hidden = None):
    print(target)
    encoder_outputs, encoder_hidden = encoder(input, encoder_hidden)
    decoder_outputs, decoder_hidden = decoder(encoder_outputs, decoder_hidden)
    print(decoder_outputs)


from sklearn.neighbors import KNeighborsClassifier
def knn(encoder, dataloader, k = 3):
    print("Collecting hidden states...")
    X = []
    y = []
    for input, _, label in dataloader:
        encoder_outputs, encoder_hidden = encoder(input, None)
        vector = encoder_hidden.squeeze().tolist()
        if dataloader.batch_size == 1:
            vector = [vector]
        
        X.extend(vector)
        y.extend(label) 

    print("Fitting KNN...")
    neigh = KNeighborsClassifier(n_neighbors= k)
    neigh.fit(X, y)
    return neigh

def predict(predictor, encoder, dataloader):
    print("Predicting...")
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
    #constructDatasetCSV("../Signals/full_dataset/")
    #dataset = SignalDataset("../Signals/full_dataset/", "dataset.csv", raw = False)
    dataset = TestDataset()
    train_size = int(0.8 * len(dataset))
    val_test_size = (len(dataset) - train_size) // 2
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,  val_test_size, val_test_size])  
 
    train(train_dataset, validation_dataset)
   
    dataloader = DataLoader(train_dataset, batch_size = 1, shuffle = True, collate_fn = collate)
    encoder = torch.load("encoder.pt")
    predictor = knn(encoder, dataloader, 3)

    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = True, collate_fn = collate)
    evaluate(test_dataloader)
    predict(predictor, encoder, test_dataloader)
    
    