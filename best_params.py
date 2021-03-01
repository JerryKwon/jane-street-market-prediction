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
        self.hidden_layer, self.n_layers, self.decreasing, self.f_act, self.dropout, self.embed_dim, self.optimizer, self.learning_rate, self.weight_decay= self.set_params(selection)

    def set_params(self, selection):
        print(type(selection))
        hidden_layer = None
        n_layers = None
        decreasing = None
        f_act = None
        dropout = None
        embed_dim = None
        optimizer = None
        learning_rate = None
        weight_decay = None

        # selection might be string
        if selection == 1:
            print("hello")
            hidden_layer = 512
            n_layers = 3
            decreasing = True
            f_act = nn.LeakyReLU()
            dropout = 0.34213845887711536
            embed_dim = 10
            optimizer = optim.Adam
            learning_rate = 0.0009437366580626903
            weight_decay = 1.0288953711004482e-08

        elif selection == 2:
            hidden_layer = 256
            n_layers = 2
            decreasing = False
            f_act = nn.SiLU()
            dropout = 0.49627361377205387
            embed_dim = 0
            optimizer = optim.Adam
            learning_rate = 1.3352033297894747e-05
            weight_decay = 8.62843672831598e-08

        return hidden_layer, n_layers, decreasing, f_act, dropout, embed_dim, optimizer, learning_rate, weight_decay

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