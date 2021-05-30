# -*- coding: utf-8 -*-
# @Time : 2021/4/7 17:14
# @Author : luff543
# @Email : luff543@gmail.com
# @File : data.py
# @Software: PyCharm

import os
import numpy as np
from pandas import read_csv
from tqdm import tqdm
import tensorflow_hub as hub
from ckiptagger import WS
from opencc import OpenCC
import math

import collections
import tensorflow as tf


import bert

class Tokenizer(bert.bert_tokenization.FullTokenizer):
    def __init__(self,vocab_file, do_lower_case,tokenizer_mod = "CKIP"):
        super().__init__(vocab_file, do_lower_case)
        self.mod = tokenizer_mod
        self.ws = WS("CKIP_data/")
    def tokenize(self, text):
        # print("use tokenize"+self.mod)
        if self.mod == "CKIP":
            # print("use tokenize" + self.mod)
            split_tokens = self.ws([text],
                                    sentence_segmentation=True,
                                    segment_delimiter_set={'?', '？', '!', '！', '。', ',',
                                                           '，', ';', ':', '、'})[0]
        else:
            split_tokens = []
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)

        return split_tokens


class BertDataManager:
    """
    Bert的數據管理器
    """

    def convert_tokens_to_ids(self,tokens):  # 输入为词表，和要转化的 text
        self.tokenizer.mod = "bert"
        id = []
        for token in tokens:
            # print(self.tokenizer.tokenize(token))
            id_temp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(token))
            id.append(sum(id_temp)/len(id_temp))

        # print(id)
        self.tokenizer.mod = "CKIP"
        return id

    def __init__(self, configs, logger):
        self.configs = configs
        self.train_file = configs.train_file
        self.logger = logger
        self.hyphen = configs.hyphen

        self.train_file = configs.datasets_fold + '/' + configs.train_file

        if configs.dev_file is not None:
            self.dev_file = configs.datasets_fold + '/' + configs.dev_file
        else:
            self.dev_file = None

        self.test_file = configs.datasets_fold + '/' + 'test.txt'

        self.label_scheme = configs.label_scheme
        self.label_level = configs.label_level
        self.suffix = configs.suffix
        self.PADDING = '[PAD]'

        self.batch_size = configs.batch_size
        self.max_sequence_length = configs.max_sequence_length
        self.label2onehot = {0:[1,0],1:[0,1]}

        hub_url = "https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/2"
        # hub_url = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2"
        l_bert = hub.KerasLayer(hub_url, trainable=True)

        vocab_file = l_bert.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = l_bert.resolved_object.do_lower_case.numpy()

        self.tokenizer = Tokenizer(vocab_file, do_lower_case,configs.tokenizer)


        self.max_token_number = len(self.tokenizer.vocab)
        self.max_label_number = 2

    def next_batch(self, X, y, att_mask, segment, start_index):
        """
        下一次個訓練批次
        :param X:
        :param y:
        :param att_mask:
        :param start_index:
        :return:
        """
        last_index = start_index + self.batch_size
        X_batch = list(X[start_index:min(last_index, len(X))])
        y_batch = list(y[start_index:min(last_index, len(X))])
        att_mask_batch = list(att_mask[start_index:min(last_index, len(X))])
        segment_batch = list(segment[start_index:min(last_index, len(X))])
        if last_index > len(X):
            left_size = last_index - (len(X))
            for i in range(left_size):
                index = np.random.randint(len(X))
                X_batch.append(X[index])
                y_batch.append(y[index])
                att_mask_batch.append(att_mask[index])
                segment_batch.append(segment[index])
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        att_mask_batch = np.array(att_mask_batch)
        segment_batch = np.array(segment_batch)
        return X_batch, y_batch, att_mask_batch, segment_batch

    def get_word(self,sentence):

        cc = OpenCC("s2twp")
        sentence = cc.convert(sentence)
        temp_tokenize = self.tokenizer.tokenize(sentence);
        temp_tokenize = [self.convert_tokens_to_ids(['[CLS]']+temp_tokenize[i:i+self.max_sequence_length-2]+['[SEP]']) for i in range(0, len(temp_tokenize), self.max_sequence_length - 2)]

        return temp_tokenize

    def prepare(self, df):
        self.logger.info('loading data...')
        X = []
        Y = []
        att_mask = []
        segment = []
        for index , (y,x) in tqdm(df.iterrows()):
            # if index>=1000:
            #     break
            if str(x) != 'nan':
                    tmp_x = self.get_word(x);
                    tmp_y = []
                    tmp_att_mask = []
                    tmp_segment = []
                    for i in range(len(tmp_x)):
                        tmp_att_mask.append([1] * len(tmp_x[i]))
                        tmp_y.append(self.label2onehot[int(y)])
                    # padding
                        tmp_x[i] += [0 for _ in range(self.max_sequence_length - len(tmp_x[i]))]
                        tmp_att_mask[i] += [0 for _ in range(self.max_sequence_length - len(tmp_att_mask[i]))]
                        tmp_segment.append([0] * self.max_sequence_length)
                    X += tmp_x
                    Y += tmp_y
                    att_mask += tmp_att_mask
                    segment += tmp_segment

        # print(len(X),len(Y),len(att_mask),len(segment))
        # print(np.array(X))
        # print(np.array(Y))
        return np.array(X), np.array(Y), np.array(att_mask), np.array(segment)

    def get_training_set(self, train_val_ratio=0.9):
        """
        獲取訓練數據集、驗證集
        :param train_val_ratio:
        :return:
        """
        df_train = read_csv(self.train_file)
        X, y, att_mask, segment = self.prepare(df_train)
        # shuffle the samples
        num_samples = len(X)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        att_mask = att_mask[indices]
        data_size = len(X)
        rate = 0.8
        split_boundary = int(data_size*rate)
        return X[:split_boundary], y[:split_boundary], att_mask[:split_boundary],  segment[:split_boundary]\
            , X[split_boundary:], y[split_boundary:], att_mask[split_boundary:],  segment[split_boundary:]

    def get_valid_set(self):
        """
        獲取驗證集
        :return:
        """
        df_val = read_csv(self.dev_file)
        X_val, y_val, att_mask_val, segment_val = self.prepare(df_val)
        return X_val, y_val, att_mask_val, segment_val

    def get_testing_set(self):
        """
        獲取驗證集
        :return:
        """
        df_test = read_csv(self.test_file)
        X_test, y_test, att_mask_test, segment_test = self.prepare(df_test)
        return X_test, y_test, att_mask_test, segment_test

    def prepare_single_sentence(self, sentence):
        """
        把預測的句子轉成矩陣和向量
        :param sentence:
        :return:
        """
        sentence = list(sentence)
        if len(sentence) <= self.max_sequence_length - 2:
            x = self.tokenizer.encode(sentence)
            att_mask = [1] * len(x)
            x += [0 for _ in range(self.max_sequence_length - len(x))]
            att_mask += [0 for _ in range(self.max_sequence_length - len(att_mask))]
        else:
            sentence = sentence[:self.max_sequence_length - 2]
            x = self.tokenizer.encode(sentence)
            att_mask = [1] * len(x)
        y = [self.label2id['O']] * self.max_sequence_length
        return np.array([x]), np.array([y]), np.array([att_mask]), np.array([sentence])
