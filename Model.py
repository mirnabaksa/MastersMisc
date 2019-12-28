import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers = 1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.RNN = nn.GRU(input_size, hidden_size,  batch_first = True)

    def forward(self, input, hidden = None):
        outputs, hidden = self.RNN(input, hidden)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers = 1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.RNN = nn.GRU(hidden_size, output_size, batch_first = True)

    def forward(self, input, hidden = None):
        output,  hidden = self.RNN(input, hidden)
        output = pad_packed_sequence(output, batch_first = True)
        output = output[0].squeeze()
        return output
