import torch.nn as nn
import torch.optim as optim
from models import ResidualBlock

class Resnet1dcnn:
    def __init__(self):
        self.block = ResidualBlock
        self.layers = [5,5,5]
        self.hidden_layers = [512, 128]
        self.optimizer = optim.Adam
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5

class ResnetLinear:
    def __init__(self, selection):
        self.set_params(selection)

    def set_params(self, selection):
        if selection == 1:
            self.hidden_layer = 512
            self.n_layers = 3
            self.decreasing = True
            self.f_act = nn.LeakyReLU()
            self.dropout = 0.34213845887711536
            self.embed_dim = 10
            self.optimizer = optim.Adam
            self.learning_rate = 0.0009437366580626903
            self.weight_decay = 1.0288953711004482e-08

        elif selection == 2:
            self.hidden_layer = 256
            self.n_layers = 2
            self.decreasing = False
            self.f_act = nn.SiLU()
            self.dropout = 0.49627361377205387
            self.embed_dim = 0
            self.optimizer = optim.Adam
            self.learning_rate = 1.3352033297894747e-05
            self.weight_decay = 8.62843672831598e-08

class EmbedNN:
    def __init__(self, selection):
        self.set_params(selection)

    def set_params(self, selection):
        if selection == 1:
            self.hidden_layer = 256
            self.n_layers = 3
            self.decreasing = False
            self.f_act = nn.SiLU()
            self.dropout = 0.23308511537027937
            self.embed_dim = 10
            self.optimizer = optim.Adam
            self.learning_rate = 0.000663767918321238
            self.weight_decay = 2.6504094565959894e-07

        elif selection == 2:
            self.hidden_layer = 256
            self.n_layers = 4
            self.decreasing = True
            self.f_act = nn.SiLU()
            self.dropout = 0.17971171427796284
            self.embed_dim = 5
            self.optimizer = optim.Adam
            self.learning_rate = 2.9521544108896628e-05
            self.weight_decay = 5.679142529741758e-05