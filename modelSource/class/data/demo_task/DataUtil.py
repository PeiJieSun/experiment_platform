'''
    The initial point, call data, generate train, val, test, test_eva, val_eva
'''
import os
from time import time
from DataModule import DataModule

class DataUtil():
    def __init__(self, conf):
        self.conf = conf

    # pmf, bpr
    def initializeTask1Handle(self):
        self.createTrainHandle()

    def createTrainHandle(self):
        data_dir = self.conf.data_dir
        train_filename = "%s.train.json" % data_dir
        val_filename = "%s.val.json" % data_dir
        test_filename = "%s.test.json" % data_dir

        self.train = DataModule(self.conf, train_filename)
        self.val = DataModule(self.conf, val_filename)
        self.test = DataModule(self.conf, test_filename)