#-*- coding:utf-8 -*-

from typing import Type, Any, Callable, Union, List, Optional

import numpy as np
import torch
import torch.nn as nn


# ResidualBlock for Resnet-1dcnn
class ResidualBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplane, plane, stride=1, dilation=1, dropout=0.2, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = self.conv3x3(inplane, plane, stride, dilation)
        self.bn1 = nn.BatchNorm1d(plane)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = self.conv3x3(plane, plane)
        self.bn2 = nn.BatchNorm1d(plane)

        # Inplace means in Activation Func
        # https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample

    def conv3x3(self, in_planes, out_planes, stride=1, dilation=1):
        return nn.Conv1d(in_planes, out_planes, 3, stride, padding=dilation, bias=False)

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut
        out = self.relu(out)

        return out

# Resnet-1dcnn
class Resnet1dcnn(nn.Module):
    def __init__(self, block: Type[ResidualBlock], layers: List[int], dropout=0.2, num_feature=130,
                 hidden_layers=[512, 256], num_classes: int = 5):
        super(Resnet1dcnn, self).__init__()
        self.inplanes = 64
        self.block = block
        self.dropout = dropout
        self.num_feature = num_feature
        self.h1, self.h2 = hidden_layers
        self.num_classes = num_classes
        self.reshaped_dim = int(self.h1 / self.inplanes)

        self.relu = nn.ReLU(inplace=False)

        self.bn_d0 = nn.BatchNorm1d(self.num_feature)
        self.dropout_d0 = nn.Dropout(self.dropout)

        self.dense1 = nn.Linear(self.num_feature, self.h1)
        self.bn_d1 = nn.BatchNorm1d(self.h1)
        self.dropout_d1 = nn.Dropout(self.dropout)

        self.layer1 = self.make_layers(self.block, 64, layers[0], stride=1)
        self.layer2 = self.make_layers(self.block, 128, layers[1], stride=2)
        self.layer3 = self.make_layers(self.block, 256, layers[2], stride=2)

        self.avgpool = nn.AvgPool1d(2)
        self.flt = nn.Flatten()

        self.dense2 = nn.Linear(int(self.h1 / 2), self.h2)
        self.bn_d2 = nn.BatchNorm1d(self.h2)
        self.dropout_d2 = nn.Dropout(self.dropout)

        self.dense3 = nn.Linear(self.h2, self.num_classes)

    def make_layers(self, block, planes, layer, stride=1):
        downsample = None

        if stride > 1:
            downsample = nn.Sequential(
                self.conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=1, dropout=self.dropout, downsample=downsample))
        self.inplanes = planes * block.expansion

        for i in range(layer - 1):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def conv1x1(self, in_planes, out_planes, stride=1):
        return nn.Conv1d(in_planes, out_planes, 1, stride=stride, bias=False)

    def forward(self, x):

        x = self.bn_d0(x)
        x = self.dropout_d0(x)

        x = self.dense1(x)
        x = self.bn_d1(x)
        x = self.relu(x)
        x = self.dropout_d1(x)

        x = x.reshape(x.size(0), 64, self.reshaped_dim)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)

        x = self.flt(x)

        x = self.dense2(x)
        x = self.bn_d2(x)
        x = self.relu(x)
        x = self.dropout_d2(x)

        x = self.dense3(x)

        return x

# ResnetLinear
class ResnetLinear(nn.Module):
    def __init__(self, num_features, num_tags, num_classes, hidden_layer, n_layers, decreasing, f_act, dropout, embed_dim, df_features, device, verbose=False):
        super(ResnetLinear, self).__init__()

        self.hidden_layer = hidden_layer
        self.num_layers = n_layers
        self.decreasing = decreasing

        self.f_act = f_act
        self.dropout = dropout

        self.embed_dim = embed_dim

        self.num_features = num_features
        self.num_classes = num_classes

        self.hidden_layers = None
        self.emb_mode = None

        if verbose:
            print("ResnetLinear Trial")
            print(
                f"hidden_layer:{self.hidden_layer}; num_layers:{self.num_layers}; decreasing:{self.decreasing}; f_act:{self.f_act}; dropout:{self.dropout}; embed_dim:{self.embed_dim}")

        if self.embed_dim == 0:
            self.emb_mode = False

        else:
            self.emb_mode = True

            # df_features tag num is 29(fixed value)
            self.n_feat_tags = num_tags

            self.device = device

            self.df_features = df_features.loc[:, df_features.columns[1:]]
            self.df_features["tag_29"] = np.array([1] + [0] * (self.df_features.shape[0] - 1))
            self.df_features = self.df_features.astype("int8")
            self.features_tag_matrix = torch.tensor(self.df_features.values).to(self.device)

            self.n_feat_tags += 1
            self.tag_embedding = nn.Embedding(self.n_feat_tags + 1, self.embed_dim)
            self.tag_weights = nn.Linear(self.n_feat_tags, 1)

        self.bn_d0 = nn.BatchNorm1d(self.num_features + self.embed_dim)

        if self.decreasing:
            self.hidden_layers = [int(self.hidden_layer / 2 ** (i)) for i in range(self.num_layers)]
        else:
            self.hidden_layers = [int(self.hidden_layer) for i in range(self.num_layers)]

        self.hidden_layers = [int(self.num_features + self.embed_dim)] + self.hidden_layers

        denses = list()

        for i in range(len(self.hidden_layers) - 1):
            if i == 0:
                denses.append(
                    self.make_layers(self.hidden_layers[i], self.hidden_layers[i + 1], self.dropout, self.f_act))
            else:
                denses.append(
                    self.make_layers(self.hidden_layers[i - 1] + self.hidden_layers[i], self.hidden_layers[i + 1],
                                     self.dropout, self.f_act))

        self.denses = nn.Sequential(*denses)

        self.out_dense = nn.Linear(self.hidden_layers[-1] + self.hidden_layers[-2], self.num_classes)

    def make_layers(self, in_channels, out_channels, dropout=None, f_act=nn.ReLU()):
        layers = list()
        layers.append(nn.Linear(in_channels, out_channels))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(f_act)

        if dropout:
            layers.append(nn.Dropout(dropout))

        module = nn.Sequential(*layers)

        return module

    # function to make embedding vector of Tag information per Features_0...129
    def features2emb(self):
        # one tag embedding to embed_dim dimension (1,embed_dim) per element
        all_tag_idxs = torch.LongTensor(np.arange(self.n_feat_tags)).to(self.device)
        tag_bools = self.features_tag_matrix
        f_emb = self.tag_embedding(all_tag_idxs).repeat(130, 1, 1)
        # f_emb에서 tag에 해당하는 값만 f_emb에 남김.
        f_emb = f_emb * tag_bools[:, :, None]

        # 각 feature 별로 먗개의 tag가 속하는가?
        s = torch.sum(tag_bools, dim=1)
        # 각 feature 별로 tag값에 해당하여 남겨진 embedding 값을 dimension 별로 합산(1,1,29) / 각 featrue별로 구해진 tag 개수와 division
        f_emb = torch.sum(f_emb, dim=-2) / s[:, None]

        return f_emb

    def forward(self, x):

        # if embedding
        if self.emb_mode:
            f_emb = self.features2emb()
            x = x.view(-1, self.num_features)
            x_emb = torch.matmul(x, f_emb)
            x = torch.hstack((x, x_emb))

        # num_features + embed_dim
        x = self.bn_d0(x)

        x_prev = None
        x_now = None

        for idx, dense in enumerate(self.denses):
            if idx == 0:
                x_prev = x
                x_now = dense(x_prev)
                x = torch.cat([x_prev, x_now], 1)
                x_prev = x_now
            else:
                x_now = dense(x)
                x = torch.cat([x_prev, x_now], 1)
                x_prev = x_now

        x5 = self.out_dense(x)

        return x5

# Feed-Forward-Network for Embed-NN
class FFN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_layers, dropout, f_act, is_op_act=False):
        super(FFN, self).__init__()

        self.bn_d0 = nn.BatchNorm1d(num_features)

        self.hidden_layers = [num_features] + hidden_layers

        denses = list()
        for i in range(len(self.hidden_layers) - 1):
            denses.append(self.make_layers(self.hidden_layers[i], self.hidden_layers[i + 1], f_act, dropout))

        self.denses = nn.Sequential(*denses)

        self.out_dense = None

        if num_classes > 0:
            self.out_dense = nn.Linear(self.hidden_layers[-1], num_classes)

        self.out_activ = None

        if is_op_act:
            if num_classes == 1 or num_classes == 2:
                self.out_active = nn.Sigmoid()
            elif num_classes > 2:
                self.out_active = nn.Softmax(dim=-1)

    def make_layers(self, in_channels, out_channels, f_act, dropout):
        layers = list()
        layers.append(nn.Linear(in_channels, out_channels))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(f_act)

        if dropout:
            layers.append(nn.Dropout(dropout))

        module = nn.Sequential(*layers)

        return module

    def forward(self, x):

        x = self.bn_d0(x)

        x = self.denses(x)

        if self.out_dense:
            x = self.out_dense(x)
        if self.out_activ:
            x = self.out_active(x)

        return x

# Embed-NN Model
class EmbedNN(nn.Module):
    def __init__(self, num_features, num_tags, num_classes, hidden_layer, n_layers, decreasing, f_act, dropout, embed_dim, df_features, device, verbose=False):
        super(EmbedNN, self).__init__()

        self.num_features = num_features
        self.n_feat_tags = num_tags
        self.num_classes = num_classes

        # self.hidden_layers = hidden_layers
        # self.embed_dim = embed_dim
        self.hidden_layer = hidden_layer
        self.num_layers = n_layers
        self.decreasing = decreasing

        if self.decreasing:
            self.hidden_layers = [int(self.hidden_layer / 2 ** (i)) for i in range(self.num_layers)]
        else:
            self.hidden_layers = [int(self.hidden_layer) for i in range(self.num_layers)]

        self.f_act = f_act
        self.dropout = dropout

        self.embed_dim = embed_dim

        self.embed_mode = None

        if verbose:
            print("Embed-NN Trial")
            print(
                f"hidden_layer:{self.hidden_layer}; num_layers:{self.num_layers}; decraesing:{self.decreasing}; f_act:{self.f_act}; dropout:{self.dropout}; embed_dim:{self.embed_dim}")

        if self.embed_dim == 0:
            self.embed_mode = False
        if self.embed_dim > 0:
            self.embed_mode = True

            self.device = device

            self.df_features = df_features.loc[:, df_features.columns[1:]]
            self.df_features["tag_29"] = np.array([1] + [0] * (self.df_features.shape[0] - 1))
            self.df_features = self.df_features.astype("int8")
            self.features_tag_matrix = torch.tensor(self.df_features.values).to(self.device)

            self.n_feat_tags += 1
            self.tag_embedding = nn.Embedding(self.n_feat_tags + 1, self.embed_dim)
            self.tag_weights = nn.Linear(self.n_feat_tags, 1)

        self.ffn = FFN(num_features=(self.num_features + self.embed_dim), num_classes=0,
                       hidden_layers=self.hidden_layers, f_act=self.f_act, dropout=self.dropout)
        self.dense = nn.Linear(self.hidden_layers[-1], self.num_classes)

    # function to make embedding vector of Tag information per Features_0...129
    def features2emb(self):
        # one tag embedding to embed_dim dimension (1,embed_dim) per element
        all_tag_idxs = torch.LongTensor(np.arange(self.n_feat_tags)).to(self.device)
        tag_bools = self.features_tag_matrix
        f_emb = self.tag_embedding(all_tag_idxs).repeat(130, 1, 1)
        # f_emb에서 tag에 해당하는 값만 f_emb에 남김.
        f_emb = f_emb * tag_bools[:, :, None]

        # 각 feature 별로 먗개의 tag가 속하는가?
        s = torch.sum(tag_bools, dim=1)
        # 각 feature 별로 tag값에 해당하여 남겨진 embedding 값을 dimension 별로 합산(1,1,29) / 각 featrue별로 구해진 tag 개수와 division
        f_emb = torch.sum(f_emb, dim=-2) / s[:, None]

        return f_emb

    def forward(self, x):
        if self.embed_mode:
            x = x.view(-1, self.num_features)
            # 130 X 5
            f_emb = self.features2emb()
            # N X 130 x 130 X 5 => N x 5 =>
            x_emb = torch.matmul(x, f_emb)
            # N X 130 + N X 5 =>
            x = torch.hstack((x, x_emb))

        x = self.ffn(x)
        x = self.dense(x)

        return x