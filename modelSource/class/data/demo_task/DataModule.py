from collections import defaultdict
import numpy as np
from time import time
import random
import torch

class DataModule():
    def __init__(self, conf, filename):
        self.conf = conf
        self.data_dict = {}
        self.terminal_flag = 1
        self.filename = filename
        self.index = 0

    '''
        This data fits for models: pmf, bpr
    '''
###########################################  Task 1 Data Preparation ############################################
    def task1InitializeData(self, model):
        self.model = model
        self.data_read_pointer = 0
        self.data_write_pointer = 0
        self.terminal_flag = 1
        self.task1ReadData()
        self.task1CreateTrainBatches()

    def task1ReadData(self):
        data_dict = {}
        with open(self.filename) as f:
            for line in f:
                line = eval(line)
                user, item, rating, idx = line['user_id'], line['item_id'], line['rating'], line['idx']
                rating = int(rating) - 1
                data_dict[idx] = [user, item, rating]
        self.data_dict = data_dict

    # this function can be regarded as the boost version of creating training batches
    def task1CreateTrainBatches(self):
        user_dict, item_dict, rating_dict = {}, {}, {}
        for (idx, record) in self.data_dict.items():
            [user, item, rating] = record
            user_dict[idx] = user
            item_dict[idx] = item
            rating_dict[idx] = rating
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.rating_dict = rating_dict

        # prepare for all the training batches
        self.data_to_be_input = {} # Which is used to store all the training batches
        
        index, total_batch_size = 0, 0
        total_batch_idx_list = list(self.data_dict.keys())
        total_length = len(total_batch_idx_list)
        while (index + self.conf.training_batch_size < total_length):
            batch_idx_list = total_batch_idx_list[index:index + self.conf.training_batch_size]
            total_batch_size += self.task1ConstructBatchBundle(batch_idx_list)
            index += self.conf.training_batch_size
            self.data_write_pointer += 1
        batch_idx_list = total_batch_idx_list[index:]
        total_batch_size += self.task1ConstructBatchBundle(batch_idx_list)
        self.data_write_pointer += 1
        print('total_batch_size: %s' % total_batch_size)

    # This function is used to construct the training batch bundle
    def task1ConstructBatchBundle(self, batch_idx_list):
        user_list, item_list, rating_list = [], [], []
        for idx in batch_idx_list:
            user_list.append(self.user_dict[idx])
            item_list.append(self.item_dict[idx])
            rating_list.append(self.rating_dict[idx])

        user_list = np.reshape(user_list, (-1))
        item_list = np.reshape(item_list, (-1))
        rating_list = np.reshape(rating_list, (-1))

        current_batch_size = len(batch_idx_list)
        self.data_to_be_input[self.data_write_pointer] = [user_list, item_list, rating_list, batch_idx_list]
        return current_batch_size

    # this function is bundled with the boost version of task1CreateTrainBatches
    def task1GetBatch(self):
        [self.user_list, self.item_list, self.rating_list, self.batch_idx_list] = \
            self.data_to_be_input[self.data_read_pointer]
        self.task1CreateMap()
        self.data_read_pointer += 1 # Important, update data read pointer

        if self.data_read_pointer == self.data_write_pointer:
            self.terminal_flag = 0
            self.data_read_pointer = 0

    def task1CreateMap(self):
        self.data_dict['USER_LIST'] = self.user_list
        self.data_dict['ITEM_LIST'] = self.item_list
        self.data_dict['RATING_LIST'] = self.rating_list