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

def collate(batch):
    max_len = max(len(x) for x in batch)
  

def train(dataset, iterations, hidden_size = 64, batch_size = 2, learning_rate = 0.01):
    dataloader = DataLoader(dataset, batch_size = batch_size)
    encoder = Encoder(1, hidden_size).to(device)
    decoder = Decoder(hidden_size, 1).to(device)

    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())
    criterion = nn.MSELoss()

    #print(x)
   # y = torch.zeros((5, 2, 1))
    #y[0] = torch.Tensor([[0.0, 0.1, 0.2, 0.3, 0.4])
    #y[1] = torch.Tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    #padded = pad_sequence(y, batch_first = False)
   # print("padded",x)

    #x = y.view(3,2,1)   
    #print(y)
    #print(x)

    #y = torch.Tensor([0.0, 0.1, 0.2])
    #x_v = y.view(len(y), 1, -1)
   # print(y)
    #print(x_v)
   # t[1] = x1
    #lens = [len(x_i) for x_i in x]
    #packed = pack_padded_sequence(pad_sequence(x, batch_first = True), lens, batch_first = True, enforce_sorted = False)
 #print(packed)
    for iter in range(250):
        loss_acc = 0
        last_hidden = None
        for i, batch in enumerate(dataloader):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
      
            encoder_outputs, encoder_hidden = encoder(batch, last_hidden)
            last_hidden = encoder_hidden
            encoded = encoder_hidden.repeat((len(batch), 1, 1))
            decoder_output = decoder(encoded, None)
      
        #decoder_output, _ = pad_packed_sequence(decoder_output)
      
            loss = criterion(decoder_output, batch.squeeze())
            loss_acc += loss
            if iter%10 == 0:
                print("iter", iter, loss)
                print(batch.squeeze(), "\n", decoder_output)
      
            loss.backward(retain_graph = True)
            encoder_optimizer.step()
            decoder_optimizer.step()
        
        if iter%10 == 0:
            print("Loss acc", loss_acc/(len(dataset)/batch_size))
        loss_acc = 0

    torch.save(encoder, "encoder.pt")
    torch.save(decoder, "decoder.pt")


def decode(input, encoder, decoder):
    print(input.squeeze())
    encoder_outputs, encoder_hidden = encoder(input, None)
    encoded = encoder_hidden.repeat((len(input), 1, 1))  
    decoder_output = decoder(encoded, None)
    print(decoder_output)

def evaluate():
    encoder = torch.load("encoder.pt")
    decoder = torch.load("decoder.pt")
    encoder.eval()
    decoder.eval()

    y = torch.Tensor([0.8, 0.2])
    x_v = y.view(len(y), 1, -1)
    decode(x_v, encoder, decoder)
    y = torch.Tensor([0.3, 0.7])
    x_v = y.view(len(y), 1, -1)
    decode(x_v, encoder, decoder)


if __name__ == '__main__':

    # constructDatasetCSV("../data_ecoli/reads/ecoli/")
    dataset = SignalDataset("../data_ecoli/reads/ecoli/", "test.csv")
    #train(dataset, 5)

    evaluate()