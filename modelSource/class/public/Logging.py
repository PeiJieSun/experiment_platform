#coding: utf8

import os, shutil
from termcolor import colored
import ConfigParser as cp

class Logging():
    def __init__(self, filename, debug_flag=1):
        self.filename = filename
        self.debug_flag = debug_flag
    
    def record(self, str_log):
        debug_flag = self.debug_flag
        if debug_flag == 1:
            filename = self.filename
            print(str_log)
            with open(filename, 'a') as f:
                f.write("%s\r\n" % str_log)
                f.flush()
        elif debug_flag == 0:
            print(str_log)

    def recordColor(self, str_log, color):
        debug_flag = self.debug_flag
        if debug_flag == 1:
            filename = self.filename
            print(colored(str_log, color))
            with open(filename, 'a') as f:
                f.write("%s\r\n" % str_log)
                f.flush()