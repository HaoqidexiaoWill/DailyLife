# coding=utf-8

import time
from sklearn import metrics
import random
import numpy as np
import logging 
import sys
import torch

def get_device(gpu_id):
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        return device
    else:
        raise ValueError("device is cpu, not recommend")
    

def get_train_logger(log_path):
    logger = logging.getLogger('train-{}'.format(__name__))
    logger.setLevel(logging.INFO)
    #控制台
    handler_stream = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler_stream)

    #日志文件
    handler_file = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    return logger