import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder
from .feature_engineering import get_features


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.8, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in self.args.cate_cols:

            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)
                a = df[col].unique().tolist()

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
            self.args.num_emb[col] = len(a)
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        return df

    def __feature_engineering(self, df):
        df = get_features(df)

        # NaN value가 있는 행 제거
        df_drop = df.dropna(axis=0)
        # NaN value가 있는 행 0으로 채움
        # df_drop = df.fillna(0)

        return df_drop

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path, parse_dates=["Timestamp"])  # , nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # data split에서 섞이지 않기 위해  continuous dataset, categories dataset를 임의로 합침
        total_cols = self.args.cate_cols + self.args.cont_cols + ["answerCode"]
        
        group = (
            df[["userID"] + total_cols]
            .groupby("userID")
            .apply(lambda r: list(r[column].values for column in total_cols)))

        return group.values

    def data_augmenation(self, group_data):
        total_data = list()
        for data in group_data:
            seq_len = len(data[0])
            if seq_len > self.args.max_seq_len:
                col_len = len(data)
                for i in range(min(self.args.num_limit, (seq_len-self.args.max_seq_len)//(self.args.max_seq_len//4))):
                    min_num = (seq_len-self.args.max_seq_len)-(i*(self.args.max_seq_len//4)) 
                    max_num = min_num + self.args.max_seq_len
                    total_data.append(list(data[j][min_num : max_num] for j in range(col_len)))
            else:
                total_data.append(data)

        return total_data          

    def load_train_data(self, file_name):
        tmp_data = self.load_data_from_file(file_name)
        self.train_data = self.data_augmenation(tmp_data)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        features = list(self.data[index])    
        seq_len = len(features[0])

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(features):
                features[i] = col[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        features.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(features):
            features[i] = torch.tensor(col)
        
        return features

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col) :] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

    return train_loader, valid_loader
