import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
import torch
import torch.nn as nn
from torch import optim
from torch.nn import MSELoss
from torch.nn.utils.rnn import  pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

import numpy as np

from util import asMinutes, timeSince, showPlot, timeNow, constructDatasetCSV, knn, visualize
from Model import Encoder, Decoder
from SignalDataset import Signal, SignalDataset, TestDataset, SOS_token, EOS_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed(1)

def collate(input):
    in_batch, labels = map(list, zip(*input))

    padded = pad_sequence(in_batch, batch_first = True, padding_value = 0)
    lens = [len(x) for x in in_batch]
    max_len = max(lens)
    out = pack_padded_sequence(padded, lens, batch_first = True, enforce_sorted = False)
    return out, padded, labels, max_len, lens
  

def train(train_dataset, validation_dataset = None, iterations = 150, hidden_size = 64, batch_size = 16):
    print("Training...")
    train = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate)
    validation = DataLoader(validation_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate)

    encoder = Encoder(1, hidden_size).to(device)
    decoder = Decoder(hidden_size, 1).to(device)

    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())
    criterion = nn.MSELoss()

    train_losses = []
    validation_losses = []

    for iter in range(iterations):
        encoder.train()
        decoder.train()
        
        loss_acc = 0
        for input_tensor, target_tensor, _, max_len, lens in train: 
            _, encoder_hidden = encoder(input_tensor, None)
            decoder_hidden = encoder_hidden
          
            decoder_input = target_tensor[:,0].view(batch_size, 1, 1)
            outputs = torch.zeros(batch_size, max_len)
        
            for di in range(1, max_len):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                outputs[:,di] = decoder_output.view(batch_size)
                decoder_input = decoder_output.detach() 

            for i in range(len(lens)):
                outputs[i,lens[i]:] = 0
            
        
            """ if iter == iterations-1:
                print(target_tensor[:,1:].squeeze())
                print(outputs[:,1:].squeeze())
                print() """

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad() 

            batch_loss = criterion(outputs[:,1:].squeeze(), target_tensor[:,1:].squeeze())
            batch_loss.backward(retain_graph = True)
            loss_acc += batch_loss.item()
         
            encoder_optimizer.step()
            decoder_optimizer.step() 
        
        train_losses.append(loss_acc)

        with torch.no_grad():
            val_loss_acc = 0
            for input_tensor, target_tensor, _, max_len, lens in validation:
                val_batch_size = len(target_tensor)
            
                _, encoder_hidden = encoder(input_tensor)
                decoder_hidden = encoder_hidden
        
                decoder_input = target_tensor[:,0].view(val_batch_size, 1, 1)
                decoder_hidden = encoder_hidden
                outputs = torch.zeros(val_batch_size, max_len)
        
                for di in range(1, max_len):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    outputs[:,di] = decoder_output.view(val_batch_size)
                    decoder_input = decoder_output
                
                for i in range(len(lens)):
                    outputs[i,lens[i]:] = 0

                val_loss = criterion(outputs[:,1:].squeeze(), target_tensor[:,1:].squeeze())
                val_loss_acc += val_loss.item()
        
            validation_losses.append(val_loss_acc) 

        if iter%1 == 0:
            print("Iteration:", iter, 
            " Train loss: ", "{0:.5f}".format(loss_acc/len(train)), 
            " Validation loss: ", "{0:.5f}".format(validation_losses[-1])
            )
        
    showPlot(train_losses, validation_losses)
    torch.save(encoder, "models/encoder.pt")
    torch.save(decoder, "models/decoder.pt")

def evaluate(test_dataloader):
    encoder = torch.load("models/encoder.pt")
    decoder = torch.load("models/decoder.pt")

    with torch.no_grad():
        for input_tensor, target_tensor, _, max_len, lens in test_dataloader:
                val_batch_size = len(target_tensor)
            
                _, encoder_hidden = encoder(input_tensor)
                decoder_hidden = encoder_hidden
        
                decoder_input = target_tensor[:,0].view(val_batch_size, 1, 1)
                decoder_hidden = encoder_hidden
                outputs = torch.zeros(val_batch_size, max_len)
        
                for di in range(1, max_len):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    outputs[:,di] = decoder_output.view(val_batch_size)
                    decoder_input = decoder_output
                
                for i in range(len(lens)):
                    outputs[i,lens[i]:] = 0

            
                print("Target")
                print(target_tensor[:,1:].squeeze())
                print("Model out")
                print(outputs[:,1:].squeeze())
                print()

def get_latent(dataloader, model):
    print("Collecting latent vector...")
    X = []
    y = []
    for input_tensor, _, label, _, _ in dataloader:
        vector = model.get_latent(input_tensor).tolist()
        if dataloader.batch_size == 1:
            vector = [vector]
     
        X.extend(vector)
        y.extend(label) 


    print("Latent X length: ", len(X))
    print("Latent Y length: ", len(y))
    return X, y


def predict(predictor, model, dataloader):
    print("Predicting...")
    correct = 0
    for X, target, labels, _, _ in dataloader:
 
        encoder_hidden = np.array(model.get_latent(X).tolist())
    
        if dataloader.batch_size == 1:
            X = [X]
        pred = predictor.predict(encoder_hidden.reshape(1, -1))
        correct += sum(pred == labels)
    
    n = len(dataloader) * dataloader.batch_size
    print("Accuracy: ", correct / n)


if __name__ == '__main__':
    #constructDatasetCSV("../Signals/full_dataset/")
    #dataset = SignalDataset("../Signals/full_dataset/", "csv/dataset.csv", raw = False)
    dataset = TestDataset()
    train_size = int(0.8 * len(dataset))
    val_test_size = (len(dataset) - train_size) // 2
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,  val_test_size, val_test_size])  
 
    train(train_dataset, validation_dataset)
    
    dataloader = DataLoader(train_dataset, batch_size = 1, shuffle = True, collate_fn = collate)
    model = torch.load("models/encoder.pt")
    
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = True, collate_fn = collate)
    evaluate(test_dataloader)
 
    X, y = get_latent(dataloader, model)
    predictor = knn(X, y, 3)
    visualize(X, y, dataset.get_distinct_labels())
    predict(predictor, model, test_dataloader)
    
    
