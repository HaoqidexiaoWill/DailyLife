import json 
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import pandas as pd
import re
import unicodedata
import itertools
import random
from data_utils import normalizeString, unicodeToAscii



def readConvsFile(conv_path,file):
    conv_file = open(os.path.join(conv_path,file))
    conv_data = json.load(conv_file)
    wikiIndex = conv_data['wikiDocumentIdx']
    if len(conv_data['whoSawDoc']) == 2:
        saw = 2
    elif conv_data['whoSawDoc'] == ['user1']:
        saw = 0
    else:
        saw = 1
    convs = []
    history = ''
    for idx, utter in enumerate(conv_data['history']):
        utter['text'] = normalizeString(utter['text'])
        line = {}
        line['wikiIdx'] = wikiIndex
        line['docIdx'] = utter['docIdx']
        line['uid'] = utter['uid']
        line['history'] = history
        line['response'] = utter['text']
        history += utter['text']+' '
        line['saw'] = saw
        convs.append(line)
    
    return convs

def saveNewConvs(read_path,save_path):
    index = 0
    for file in os.listdir(read_path):
        if file.split('.')[1] != 'json':
            continue
        convs = readConvsFile(read_path,file)
        new_file = 'train'+str(index)+'.csv'
        data_file = os.path.join(save_path,new_file)
        print('Writing to new formatted line...',index)
        df = pd.DataFrame(convs)
        df.to_csv(data_file,encoding='utf-8',sep='\t')
        index += 1
        
if __name__ == '__main__':
    read_path = '../Conversations/train'
    save_path = '../Conversations/seq+att/train'
    if not os.path.exists(save_path): os.mkdir(save_path)
    saveNewConvs(read_path,save_path) 
    read_path = '../Conversations/valid'
    save_path = '../Conversations/seq+att/valid'
    if not os.path.exists(read_path): os.mkdir(read_path)
    if not os.path.exists(save_path): os.mkdir(save_path)
    saveNewConvs(read_path,save_path) 
    read_path = '../Conversations/test'
    save_path = '../Conversations/seq+att/test'
    if not os.path.exists(save_path): os.mkdir(save_path)
    saveNewConvs(read_path,save_path) 
