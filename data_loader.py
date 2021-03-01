import io
import re
import os
import platform
import warnings
import zipfile

import numpy as np
import pandas as pd
import datatable as dt

import torch

class DataLoader:

    def __init__(self):

        warnings.filterwarnings("ignore")

        # https: // stackoverflow.com / questions / 8220108 / how - do - i - check - the - operating - system - in -python / 8220141
        # https://stackoverflow.com/questions/40416072/reading-file-using-relative-path-in-python-project
        self.os_env= platform.system()

        if self.os_env == 'Linux':
            self.home_dir = 'C:\\Users\\'
            self.username = os.popen("echo %username%").read().replace('\n','')
        elif self.os_env == 'Windows':
            self.home_dir = 'C:\\Users\\'
            self.username = os.popen("echo %username%").read().replace('\n','')

        self.input_path = os.path.join(os.path.abspath(os.path.dirname(os.path.abspath('__file__'))), 'input')
        self.data_path = os.path.join(self.input_path, 'data')
        self.jane_path = os.path.join(self.data_path,'competitions','jane-street-market-prediction')

        self.output_path = os.path.join(os.path.abspath(os.path.dirname(os.path.abspath('__file__'))), 'output')
        self.model_path = os.path.join(self.output_path, "model")
        self.result_path = os.path.join(self.output_path, "result")

        # directory check
        self.check_dirs()

    def check_dirs(self):
        self.check_dir(self.input_path)
        self.check_dir(self.data_path)

        self.check_dir(self.output_path)
        self.check_dir(self.model_path)
        self.check_dir(self.result_path)

        print("directory check is done.")

    def check_dir(self, dir_path):
        if os.path.isdir(dir_path):
            return
        else:
            os.mkdir(dir_path)

    def install_kaggle(self):
        # for installing kaggle package
        os.system('pip install kaggle')
        if os.path.isfile(os.path.join(self.input_path, 'kaggle.json')):
            kaggle_path = os.path.join(self.home_dir, self.username, '.kaggle')
            json_path = os.path.join(self.input_path, 'kaggle.json')
            if self.os_env == "Windows":
                os.system(f'copy {json_path} {kaggle_path}')
                os.system(f"kaggle config set -n path -v {self.data_path}")
            print("kaggle json setting complete")

            if os.path.isdir(os.path.join(self.data_path,'competitions')) == False:
                # download dataset
                os.system(f"kaggle competitions download -c jane-street-market-prediction")
                zip_target = os.path.join(self.jane_path,'jane-street-market-prediction.zip')
                zip_file = zipfile.ZipFile(zip_target)
                zip_file.extractall(self.jane_path)

                # install custom package(Linux Only)
                if self.os_env == "Linux":
                    os.system(f"rpm -ivh {os.path.join(self.jane_path,'janestreet','competition.cpython-37m-x86_64-linux-gnu.so')}")

        else:
            raise Exception("please locate kaggle.json file for downloading dataset")

    def load_data(self):

        if ~os.path.isdir(self.jane_path):
            self.install_kaggle()

        train_file = os.path.join(self.jane_path, 'train.csv')
        features_file = os.path.join(self.jane_path, 'features.csv')
        example_test_file = os.path.join(self.jane_path, 'example_test.csv')
        example_sample_submission_file = os.path.join(self.jane_path, 'example_sample_submission.csv')

        train_data_datatable = dt.fread(train_file)

        df_train = train_data_datatable.to_pandas()
        df_features = pd.read_csv(features_file)
        df_example_test = pd.read_csv(example_test_file)
        df_example_sample_submission = pd.read_csv(example_sample_submission_file)

        return df_train, df_features, df_example_test, df_example_sample_submission

    def save_model(self, model_state, model_path):

        torch.save(model_state, model_path)

    def load_model(self, model, model_name):

        model_files = os.listdir(self.model_path)
        model_files_path = [ os.path.join(self.model_path, model_file) for model_file in model_files if model_name in model_file]

        models = []
        for model_file_path in model_files_path:
            model.load_state_dict(torch.load(model_file_path))
            models.append(model)

        return models