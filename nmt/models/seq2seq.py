import torch
import torch.nn as nn


class Encoder(nn.modules):
    def __init__(self, wordvec_dim, hidden_size, n_layers = 4, ):
        super(self).__init__()