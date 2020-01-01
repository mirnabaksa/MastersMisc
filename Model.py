import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers = 1):
        super(AutoEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.encoder = nn.GRU(input_size, hidden_size,  batch_first = True)
        self.decoder = nn.GRU(hidden_size, input_size, batch_first = True)
        self.enc_last_hidden = None
        self.dec_last_hidden = None

    def forward(self, input):
        encoder_outputs, encoder_hidden = self.encoder(input, self.enc_last_hidden)
        self.enc_last_hidden = encoder_hidden

        decoder_output, decoder_hidden = self.decoder(encoder_outputs, self.dec_last_hidden)
        self.dec_last_hidden = decoder_hidden
        output = pad_packed_sequence(decoder_output, batch_first = True)
        output = output[0].squeeze()
        return output

    def get_latent(self, input):
        encoder_outputs, encoder_hidden = self.encoder(input, None)
        return encoder_hidden

'''
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
        return output, hidden
'''

class TripletEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers = 1):
        super(TripletEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.encoder = nn.GRU(input_size, hidden_size,  batch_first = True)

    def forward(self, a, p, n, hidden = None):
        _, hidden_a = self.encoder(a, hidden)
        _, hidden_p = self.encoder(p, hidden)
        _, hidden_n = self.encoder(n, hidden)
        return hidden_a, hidden_p, hidden_n

    def get_latent(self, input):
        _, hidden = self.encoder(input, None)
        return hidden
