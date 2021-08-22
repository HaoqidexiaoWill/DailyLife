import json 
import os
import csv
import pandas as pd
import re
import torch
import unicodedata
import itertools
import random
import config


# Default word tokens
PAD_token = config.PAD_token  # Used for padding short sentences
SOS_token = config.SOS_token  # Start-of-sentence token
EOS_token = config.EOS_token  # End-of-sentence token
UNK_token = config.UNK_token  # Unkonw token


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"can't", r"can not", s)
    s = re.sub(r"n't", r" not", s)
    s = re.sub(r"'ve'", r" have", s)
    s = re.sub(r"cannot", r"can not", s)
    s = re.sub(r"what's", r"what is", s)
    s = re.sub(r"that's",r"that is",s)
    s = re.sub(r"'re", r" are", s)
    s = re.sub(r"'d", r" would", s)
    s = re.sub(r"'ll'", r" will", s)
    s = re.sub(r" im ", r" i am ", s)
    s = re.sub(r"'m", r" am", s)
    s = re.sub(r"([.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z.!?0-9]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def history2ids(voc, sentence):
    ids = []
    oovs = []
    for w in sentence.split(' '):
        if w not in voc.word2index:
            if w not in oovs:
                oovs.append(w)
            oov_num = oovs.index(w)
            ids.append(voc.num_words+oov_num)
        else:
            ids.append(voc.word2index[w])
    ids.append(EOS_token)
    return ids, oovs 

def response2ids(voc, sentence, history_oovs):
    ids = []
    for w in sentence.split(' '):
        if w not in voc.word2index:
            if w in history_oovs:
                oov_num = history_oovs.index(w)
                ids.append(voc.num_words+oov_num)
            else:
                ids.append(UNK_token)
        else:
            ids.append(voc.word2index[w])
    ids.append(EOS_token)
    return ids 

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] if word in voc.word2index else UNK_token for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    max_oov_length = 0
    ids_batch = []
    extend_ids_batch = []
    batch_oovs = []
    for sentence in l:
        ids, oov = history2ids(voc,sentence) ## with oov
        ids_batch.append(indexesFromSentence(voc,sentence)) ## without oov
        extend_ids_batch.append(ids)
        batch_oovs.append(oov)
        max_oov_length = max(max_oov_length,len(oov))
    lengths = torch.tensor([len(ids) for ids in ids_batch])
    padList = zeroPadding(ids_batch)
    padVar = torch.LongTensor(padList)
    extend_padList = zeroPadding(extend_ids_batch)
    extend_padVar = torch.LongTensor(extend_padList)
    return padVar, lengths, batch_oovs, max_oov_length,extend_padVar

def encVar(docs, voc): #(B, 3)
    ids_batch = [indexesFromSentence(voc, section)[:config.MAX_SEC_LENGTH] for doc in docs for section in doc]
    lengths = torch.tensor([len(ids) for ids in ids_batch]) #(B*3,)
    padList = zeroPadding(ids_batch) #(L,B*3)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc, oovs):
    ids_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    extend_ids_batch = []
    for i,sentence in enumerate(l):
        ids = response2ids(voc,sentence,oovs[i])
        extend_ids_batch.append(ids)
    max_target_len = max([len(ids) for ids in ids_batch])
    padList = zeroPadding(ids_batch)
    extend_padList = zeroPadding(extend_ids_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    extend_padVar = torch.LongTensor(extend_padList)
    return padVar, mask, max_target_len, extend_padVar

                                       