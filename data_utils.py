import os
import random

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from numba import njit

from data_loader import DataLoader as CustomDataLoader
from cv import StratifiedGroupKFold, GroupTimeSeriesSplit

from best_params import Resnet1dcnn as R1dcnpram
from best_params import ResnetLinear as Rlinpram
from best_params import EmbedNN as Embnnpram

from models import ResidualBlock, CustomResNet, ResnetLinear, Emb_NN_Model

class JaneDataset(Dataset):
    def __init__(self, np_X, np_y):
        super(JaneDataset, self).__init__()
        self.X = np_X
        self.y = np_y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        # date, weight, resp
        X_util = self.X[index, :3]
        X = torch.tensor(self.X[index, 3:], dtype=torch.float)
        y = torch.tensor(self.y[index], dtype=torch.float)
        return X_util, X, y

class DataUtils:
    def __init__(self, entire_seed=1029):
        self.entire_seed = entire_seed

    def prepare_data(self, cv_type="SGCV"):
        self.data_loader = CustomDataLoader()
        self.df_train, self.df_features, self.df_example_test, self.df_example_sample_submission = self.data_loader.load_data()

        self.seed_torch(self.entire_seed)

        features = [col for col in self.df_train.columns if "feature" in col]
        resps = [col for col in self.df_train.columns if "resp" in col]
        target_resp = [resp_ for resp_ in resps if "_" not in resp_]
        target = ["weight"] + target_resp + features

        df_train = self.reduce_memory_usage(self.df_train)

        # drop before 85days
        df_train = df_train.loc[df_train.date > 85]
        # drop weight 0 for training
        df_train = df_train.loc[df_train.weight > 0]

        # converting numpy for efficient calcualtion.
        # ft 1~129
        np_ft_train = df_train.loc[:, features[1:]].values
        np_ft_train.shape

        f_mean = np.nanmean(np_ft_train, axis=0)

        np_train = df_train.values

        print('fillna_npwhere_njit (mean-filling):')
        np_train[:, 8:-1] = self.for_loop(self.fillna_npwhere_njit, np_train[:, 8:-1], f_mean)

        dict_features = {col: idx for idx, col in enumerate(df_train.columns.tolist())}

        np_d_w = np_train[:, :2]
        # ['resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp']
        idx_resps = list()
        for resp in resps:
            idx_col = dict_features[resp]
            idx_resps.append(idx_col)
        np_resps = np_train[:, idx_resps]

        resps_prcntls = [50, 49, 49, 50, 50]
        resps_prcntls = [np.percentile(np_resps[:, idx], prcntls) for idx, prcntls in enumerate(resps_prcntls)]

        list_resps = list()
        for idx, resps_prcntl in enumerate(resps_prcntls):
            result = list(map(lambda x: 1 if x > resps_prcntl else 0, np_resps[:, idx]))
            list_resps.append(result)
        np_targets = np.stack(list_resps).T

        idx_target = [("resp_" not in key) and ("ts_" not in key) for key in dict_features.keys()]
        idx_target = np.arange(np_train.shape[1])[idx_target]
        X_np_train = np_train[:, idx_target]

        X = X_np_train
        y = np_targets

        # cv_type = ["SGCV","GTCV","random"]
        if cv_type == "SGCV":
            cv = StratifiedGroupKFold(n_splits=3,random_state=self.entire_seed)

            cv_idxes = [(train_idx, test_idx) for train_idx, test_idx in cv.split(X, y[:, -1], group=X[:, 0])]
            for idx, cv_idx in enumerate(cv_idxes):
                train_idx, test_idx = cv_idx
                train_dates = np.unique(X[train_idx, 0])
                test_dates = np.unique(X[test_idx, 0])


        elif cv_type == "GTCV":
            cv = GroupTimeSeriesSplit(n_splits=3)

            cv_idxes = [(train_idx, test_idx) for train_idx, test_idx in cv.split(X, y[:, -1], group=X[:, 0])]
            for idx, cv_idx in enumerate(cv_idxes):
                train_idx, test_idx = cv_idx
                train_dates = np.unique(X[train_idx, 0])
                test_dates = np.unique(X[test_idx, 0])

        elif cv_type == "random":
            dataset = JaneDataset(X, y)

            train_size = int(len(dataset) * 0.8)
            valid_size = len(dataset) - train_size

            train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(self.entire_seed))
            cv_idxes = (train_dataset, valid_dataset)

        return X,y, cv_idxes

    def train(self, model_type, X, y, cv_idxes, selection=None):

        model, optimizer, learning_rate, weight_decay = self.set_params(model_type, selection)


    def set_params(self, model_type, selection):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if model_type == "Resnet1dcnn":
            best_params = R1dcnpram()

            block = best_params.block
            hidden_layers = best_params.hidden_layers
            layers = best_params.layers
            optimizer = best_params.optimizer
            learning_rate = best_params.learning_rate
            weight_decay = best_params.weight_decay

            model = CustomResNet(block=block, layers=layers, hidden_layers=hidden_layers)

        elif model_type == "ResnetLinear":
            best_params = Rlinpram(selection=selection)

            hidden_layer = best_params.hidden_layer
            n_layers = best_params.n_layers
            decreasing = best_params.decreasing
            f_act = best_params.f_act
            dropout = best_params.dropout
            embed_dim = best_params.embed_dim
            optimizer = best_params.optimizer
            learning_rate = best_params.learning_rate
            weight_decay = best_params.weight_decay

            model = ResnetLinear(num_features=130, num_classes=5, hidden_layer=hidden_layer, n_layers=n_layers, decreasing=decreasing, f_act=f_act, dropout=dropout, embed_dim=embed_dim, df_features=self.df_features, device=device)

        elif model_type == "EmbedNN":
            best_params = Embnnpram(selection=selection)

            hidden_layer = best_params.hidden_layer
            n_layers = best_params.n_layers
            decreasing = best_params.decreasing
            f_act = best_params.f_act
            dropout = best_params.dropout
            embed_dim = best_params.embed_dim
            optimizer = best_params.optimizer
            learning_rate = best_params.learning_rate
            weight_decay = best_params.weight_decay

            model = Emb_NN_Model(self, num_features=130, num_tags=29, num_classes=5, hidden_layer=hidden_layer, n_layers=n_layers, decreasing=decreasing, f_act=f_act, dropout=dropout, embed_dim=embed_dim, df_features=self.df_features, device=device)

        return model, optimizer, learning_rate, weight_decay

    def seed_torch(self, seed=1029):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    #    torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.deterministic = False

    """
    Reduce Memory Usage by 75%
    https://www.kaggle.com/tomwarrens/nan-values-depending-on-time-of-day
    """
    ## Reduce Memory

    def reduce_memory_usage(self, df):

        start_memory = df.memory_usage().sum() / 1024 ** 2
        print(f"Memory usage of dataframe is {start_memory} MB")

        for col in df.columns:
            col_type = df[col].dtype

            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

                else:
                    #                 reducing float16 for calculating numpy.nanmean
                    #                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    #                     df[col] = df[col].astype(np.float16)
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        pass
            else:
                df[col] = df[col].astype('category')

        end_memory = df.memory_usage().sum() / 1024 ** 2
        print(f"Memory usage of dataframe after reduction {end_memory} MB")
        print(f"Reduced by {100 * (start_memory - end_memory) / start_memory} % ")
        return df

    """
    The codes from 'Optimise Speed of Filling-NaN Function'
    https://www.kaggle.com/gogo827jz/optimise-speed-of-filling-nan-function
    """

    def for_loop(self, method, matrix, values):
        for i in range(matrix.shape[0]):
            matrix[i] = method(matrix[i], values)
        return matrix

    def for_loop_ffill(self, method, matrix):
        tmp = np.zeros(matrix.shape[1], dtype=np.float32)
        for i in range(matrix.shape[0]):
            matrix[i] = method(matrix[i], tmp)
            tmp = matrix[i]
        return matrix

    @njit
    def fillna_npwhere_njit(self, array, values):
        if np.isnan(array.sum()):
            array = np.where(np.isnan(array), values, array)
        return array