import os
import random
from tqdm import tqdm

import numpy as np

from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from numba import njit

from data_loader import DataLoader as CustomDataLoader
from cv import StratifiedGroupKFold, GroupTimeSeriesSplit

from best_params import Resnet1dcnn as R1dcnpram
from best_params import ResnetLinear as Rlinpram
from best_params import EmbedNN as Embnnpram

from models import ResidualBlock, Resnet1dcnn, ResnetLinear, EmbedNN

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

class EarlyStopping:
    def __init__(self, data_loader, patience=7, mode="max", delta=0.001):
        self.data_loader = data_loader

        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score: #  + self.delta
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # ema.apply_shadow()
            self.save_checkpoint(epoch_score, model, model_path)
            # ema.restore()
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(f"Validation score improved ({self.val_score:.4f} --> {epoch_score:.4f}). Saving model!")
            # if not DEBUG:
            self.data_loader.save_model(model.state_dict(),model_path)
            # torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

class DataUtils:
    def __init__(self, entire_seed=1029):
        self.entire_seed = entire_seed
        self.data_loader = CustomDataLoader()

    def prepare_data(self, cv_type="SGCV"):

        self.df_train, self.df_features, self.df_example_test, self.df_example_sample_submission = self.data_loader.load_data()

        self.seed_torch(self.entire_seed)

        self.features = [col for col in self.df_train.columns if "feature" in col]
        resps = [col for col in self.df_train.columns if "resp" in col]
        target_resp = [resp_ for resp_ in resps if "_" not in resp_]
        target = ["weight"] + target_resp + self.features

        df_train = self.reduce_memory_usage(self.df_train)

        # drop before 85days
        df_train = df_train.loc[df_train.date > 85]
        # drop weight 0 for training
        df_train = df_train.loc[df_train.weight > 0]

        # converting numpy for efficient calcualtion.
        # ft 1~129
        np_ft_train = df_train.loc[:, self.features[1:]].values
        np_ft_train.shape

        self.f_mean = np.nanmean(np_ft_train, axis=0)

        np_train = df_train.values

        print('fillna_npwhere_njit (mean-filling):')
        np_train[:, 8:-1] = for_loop(fillna_npwhere_njit, np_train[:, 8:-1], self.f_mean)

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

    def train(self, model_type, X, y, cv_idxes, selection=None, epochs=100, batch_size=4096, early_stopping=True):

        if (model_type == "Resnet1dcnn") and (selection is not None):
            raise Exception("Resnet1dcnn is already set default params")
        elif (model_type != "Resnet1dcnn") and (selection is None):
            raise Exception("ResnetLinear and EmbedNN need to set params between 1 and 2")

        model, optimizer, learning_rate, weight_decay = self.set_params(model_type, selection)

        if model_type == "Resnet1dcnn":
            self.train_resnet1dcnn(model, X, y, cv_idxes, optimizer, learning_rate, weight_decay, epochs, batch_size, early_stopping)

        else:
            self.train_remainder(model, X, y, cv_idxes, optimizer, learning_rate, weight_decay, epochs, batch_size, early_stopping)

        return model

    def csv_inference(self, model, model_type):
        models = self.data_loader.load_model(model, model_type)

        for idx,row in tqdm(self.df_example_test.iterrows()):
            if row["weight"].item() > 0:
                test_np = row.loc[self.features].values[np.newaxis,:]
                test_np[:, 1:] = for_loop(fillna_npwhere_njit, test_np[:,1:],self.f_mean)
                results = []
                for model in models:
                    torch.cuda.empty_cache()
                    model.eval()
                    result = model(torch.tensor(test_np,dtype=torch.float).to(self.device)).detach().sigmoid().cpu().numpy()[:,-1].item()
                    results.append(result)
                pred = np.mean(results)
                action = 1 if pred >= .5 else 0
                self.df_example_sample_submission.loc[idx,"action"] = action
            else:
                self.df_example_sample_submission.loc[idx,"action"] = 0
        return self.df_example_sample_submission

    def package_inference(self, model, model_type):
        import janestreet

        env = janestreet.make_env()
        models = self.data_loader.load_model(model, model_type)

        for (test_df, pred_df) in tqdm(env.iter_test()):
            if test_df["weight"].item() > 0:
                test_np = test_df.loc[:, self.features].values
                test_np[:, 1:] = self.for_loop(self.fillna_npwhere_njit, test_np[:,1:],self.f_mean)
                results = []
                for model in models:
                    torch.cuda.empty_cache()
                    model.eval()
                    result = model(torch.tensor(test_np,dtype=torch.float).to(self.device)).detach().sigmoid().cpu().numpy()[:,-1].item()
                    results.append(result)
                pred = np.mean(results)
                action = 1 if pred >= .5 else 0
                pred_df.action = action
            else:
                pred_df.action = 0
            env.predict(pred_df)

    def train_resnet1dcnn(self, model, X, y, cv_idxes, optimizer, learning_rate, weight_decay, epochs, batch_size, patience=7):
        model_home = self.data_loader.model_path
        model_name = model.__class__.__name__

        train_dataset, valid_dataset = cv_idxes
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        self.seed_torch(self.entire_seed)
        model = model.to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        es = None
        if patience > 0:
            es = EarlyStopping(self.data_loader, patience, mode="max")
        for epoch in tqdm(range(epochs)):

            running_loss = 0.0
            running_acc = 0.0
            running_auc = 0.0
            running_util = 0.0

            model.train()

            for idx, (X_utils,inputs, labels) in enumerate(train_dataloader):
                with torch.autograd.set_detect_anomaly(True):
                    optimizer.zero_grad()
                    X_d_w = X_utils[:,:-1].detach().cpu().numpy()
                    X_r = X_utils[:,-1].detach().cpu().numpy()

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(inputs)

                    true = labels.detach().cpu().numpy()[:, -1]
                    target = np.array(
                        list(map(lambda x: 1 if x > 0.5 else 0, outputs.sigmoid().detach().cpu().numpy()[:, -1])),
                        dtype=np.float)

                    acc = (true == target).sum() / outputs.shape[0]
                    auc = roc_auc_score(true, outputs.detach().cpu().numpy()[:, -1])
                    util = utility_score(X_d_w, X_r, target)

                    running_acc += acc
                    running_auc += auc
                    running_util += util

                    loss = criterion(outputs, labels)
                    running_loss += loss.detach().item() * inputs.size(0)
                    loss.backward()
                    optimizer.step()

            epoch_loss = running_loss / len(train_dataloader.dataset)
            epoch_acc = running_acc / len(train_dataloader)
            epoch_auc = running_auc / len(train_dataloader)
            epoch_util = running_util

            model.eval()

            with torch.no_grad():

                running_loss = 0.0
                running_acc = 0.0
                running_auc = 0.0
                running_util = 0.0

                for idx, (X_utils,inputs, labels) in enumerate(valid_dataloader):
                    X_d_w = X_utils[:, :-1].detach().cpu().numpy()
                    X_r = X_utils[:, -1].detach().cpu().numpy()

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(inputs)

                    true = labels.detach().cpu().numpy()[:, -1]
                    target = np.array(
                        list(map(lambda x: 1 if x > 0.5 else 0, outputs.sigmoid().detach().cpu().numpy()[:, -1])),
                        dtype=np.float)

                    acc = (true == target).sum() / outputs.shape[0]
                    auc = roc_auc_score(true, outputs.detach().cpu().numpy()[:, -1])
                    util = utility_score(X_d_w, X_r, target)

                    running_acc += acc
                    running_auc += auc
                    running_util += util

                    loss = criterion(outputs, labels)
                    running_loss += loss.detach().item() * inputs.size(0)

                valid_loss = running_loss / len(valid_dataloader.dataset)
                valid_acc = running_acc / len(valid_dataloader)
                valid_auc = running_auc / len(valid_dataloader)
                valid_util = running_util

            print(f"EPOCH:{epoch+1}|{epochs}; loss(train/valid):{epoch_loss:.4f}/{valid_loss:.4f}; acc(train/valid):{epoch_acc:.4f}/{valid_acc:.4f}; auc(train/valid):{epoch_auc:.4f}/{valid_auc:.4f}; utility(train/valid):{epoch_util:.4f}/{valid_util:.4f}")

            model_weights = os.path.join(model_home, f"{model_name}.pth")
            if patience > 0:
                es(valid_auc, model, model_path=model_weights)
                if es.early_stop:
                    print("Early stopping")
                    break

        self.data_loader.save_model(model.state_dict(), model_weights)

    def train_remainder(self, model, X, y, cv_idxes, optimizer, learning_rate, weight_decay, epochs, batch_size, patience=7):
        model_home = self.data_loader.model_path
        model_name = model.__class__.__name__

        for _fold, cv_idx in enumerate(cv_idxes):
            print(f'Fold{_fold}:'+'.'*20)

            train_idx, valid_idx = cv_idx
            X_train = X[train_idx, :]
            y_train = y[train_idx, :]

            X_valid = X[valid_idx, :]
            y_valid = y[valid_idx, :]

            train_dataset = JaneDataset(X_train, y_train)
            valid_dataset = JaneDataset(X_valid, y_valid)

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

            self.seed_torch(seed=self.entire_seed)
            torch.cuda.empty_cache()

            model = model.to(self.device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            es = None
            if patience > 0:
                es=EarlyStopping(patience,mode="max")
            for epoch in tqdm(range(epochs)):

                running_loss = 0.0
                running_acc = 0.0
                running_auc = 0.0
                running_util = 0.0

                model.train()

                for idx, (X_utils,inputs, labels) in enumerate(train_dataloader):
                    with torch.autograd.set_detect_anomaly(True):
                        optimizer.zero_grad()
                        X_d_w = X_utils[:, :-1].detach().cpu().numpy()
                        X_r = X_utils[:, -1].detach().cpu().numpy()

                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        outputs = model(inputs)

                        true = labels.detach().cpu().numpy()[:, -1]
                        target = np.array(
                            list(map(lambda x: 1 if x > 0.5 else 0, outputs.sigmoid().detach().cpu().numpy()[:, -1])),
                            dtype=np.float)

                        acc = (true == target).sum() / outputs.shape[0]
                        auc = roc_auc_score(true, outputs.detach().cpu().numpy()[:, -1])
                        util = utility_score(X_d_w, X_r, target)

                        running_acc += acc
                        running_auc += auc
                        running_util += util

                        loss = criterion(outputs, labels)
                        running_loss += loss.detach().item() * inputs.size(0)
                        loss.backward()
                        optimizer.step()

                epoch_loss = running_loss / len(train_dataloader.dataset)
                epoch_acc = running_acc / len(train_dataloader)
                epoch_auc = running_auc / len(train_dataloader)
                epoch_util = running_util

                model.eval()

                with torch.no_grad():

                    running_loss = 0.0
                    running_acc = 0.0
                    running_auc = 0.0
                    for idx, (X_utils, inputs, labels) in enumerate(valid_dataloader):

                        X_d_w = X_utils[:,:-1].detach().cpu().numpy()
                        X_r = X_utils[:,-1].detach().cpu().numpy()

                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        outputs = model(inputs)

                        true = labels.detach().cpu().numpy()[:, -1]
                        target = np.array(
                            list(map(lambda x: 1 if x > 0.5 else 0, outputs.sigmoid().detach().cpu().numpy()[:, -1])),
                            dtype=np.float)

                        acc = (true == target).sum() / outputs.shape[0]
                        auc = roc_auc_score(true, outputs.detach().cpu().numpy()[:, -1])
                        util = utility_score(X_d_w,X_r,target)

                        running_acc += acc
                        running_auc += auc
                        running_util += util

                        loss = criterion(outputs, labels)
                        running_loss += loss.detach().item() * inputs.size(0)

                    valid_loss = running_loss / len(valid_dataloader.dataset)
                    valid_acc = running_acc / len(valid_dataloader)
                    valid_auc = running_auc / len(valid_dataloader)
                    valid_util = running_util

                print(f"EPOCH:{epoch+1}|{epochs}; loss(train/valid):{epoch_loss:.4f}/{valid_loss:.4f}; acc(train/valid):{epoch_acc:.4f}/{valid_acc:.4f}; auc(train/valid):{epoch_auc:.4f}/{valid_auc:.4f}; utility(train/valid):{epoch_util:.4f}/{valid_util:.4f}")

                model_weights = os.path.join(model_home, f"{model_name}_{_fold}.pth")
                if patience > 0:
                    es(valid_auc, model, model_path=model_weights)
                    if es.early_stop:
                        print("Early stopping")
                        break

            self.data_loader.save_model(model.state_dict(),model_weights)

    def set_params(self, model_type, selection):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = None

        if model_type == "Resnet1dcnn":
            best_params = R1dcnpram()

            block = best_params.block
            hidden_layers = best_params.hidden_layers
            layers = best_params.layers
            optimizer = best_params.optimizer
            learning_rate = best_params.learning_rate
            weight_decay = best_params.weight_decay

            model = Resnet1dcnn(block=block, layers=layers, hidden_layers=hidden_layers)

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

            model = ResnetLinear(num_features=130, num_tags=29, num_classes=5, hidden_layer=hidden_layer, n_layers=n_layers, decreasing=decreasing, f_act=f_act, dropout=dropout, embed_dim=embed_dim, df_features=self.df_features, device=self.device)

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

            model = EmbedNN(self, num_features=130, num_tags=29, num_classes=5, hidden_layer=hidden_layer, n_layers=n_layers, decreasing=decreasing, f_act=f_act, dropout=dropout, embed_dim=embed_dim, df_features=self.df_features, device=self.device)

        model = model.to(self.device)

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

    def save_csv(self, df_submission, fname):
        submission_path = os.path.join(self.data_loader.result_path,fname)
        df_submission.to_csv(submission_path)

"""
The codes from 'Optimise Speed of Filling-NaN Function'
https://www.kaggle.com/gogo827jz/optimise-speed-of-filling-nan-function
"""

def for_loop(method, matrix, values):
    for i in range(matrix.shape[0]):
        matrix[i] = method(matrix[i], values)
    return matrix

def for_loop_ffill(method, matrix):
    tmp = np.zeros(matrix.shape[1], dtype=np.float32)
    for i in range(matrix.shape[0]):
        matrix[i] = method(matrix[i], tmp)
        tmp = matrix[i]
    return matrix

@njit
def fillna_npwhere_njit(array, values):
    if np.isnan(array.sum()):
        array = np.where(np.isnan(array), values, array)
    return array


def utility_score(X_d_w, X_r, y):
    # X for date, weight, resp numpy.array
    # y for binary action by random threshold or prediction

    # date
    #     date_min, date_max = np.min(X_d_w[:,0]), np.max(X_d_w[:,0])
    unq_dates = np.unique(X_d_w[:, 0])
    period = len(unq_dates)
    #     dates = np.arange(date_min, date_max+1)

    list_p = list()

    for date in unq_dates:
        idx_date = X_d_w[:, 0] == date
        X_d = X_d_w[idx_date, 0]
        y_d = y[idx_date]
        w_d = X_d_w[idx_date, 1]
        r_d = X_r[idx_date]

        p_d = w_d * r_d * y_d
        p = p_d.sum()

        list_p.append(p)

    np_p = np.array(list_p)

    t = np.sum(np_p) / np.sqrt(np.sum(np.power(np_p, 2))) * np.sqrt(250 / period)
    utility_score = min(max(t, 0), 6) * np_p.sum()
    return utility_score