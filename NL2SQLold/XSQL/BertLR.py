import csv
import os
import sys
import re
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d,BatchNorm1d,MaxPool1d,ReLU,Dropout
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder
from torch.optim import optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
from tqdm import tqdm_notebook, trange,tqdm
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from model_dir.modeling import BertForQuestionAnswering, BertConfig,BertModel
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from Annotation import load_dataset,get_args
device = torch.device("cuda", 1)

class DataPrecessForSingleSentence(object):
    # 文本预处理
    def __init__(self,bert_tokenizer,max_workers = 10):
        self.bert_tokenizer = bert_tokenizer
        # 创建多线程池
        self.pool = ThreadPoolExecutor(max_workers=max_workers)


    def get_input(self,dataset,max_seq_len = 64):
        # 多线程对文本进行预处理
        # 输出是 [CLS][seq][SEP],seq_mask,seg_ment,labels
        sentences = dataset.iloc[:, 0].tolist()
        labels = dataset.iloc[:, 1].tolist()
        # 分词
        tokens_seq = list(self.pool.map(self.bert_tokenizer.tokenize,sentences))
        result = list(self.pool.map(self.trunate_and_pad, tokens_seq,[max_seq_len] * len(tokens_seq)))

        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        
        return seqs,seq_masks,seq_segments,labels

    def trunate_and_pad(self, seq, max_seq_len):
        # 对文本进行截断和补长
        # 输出 seq,seq_mask,seq_segment
        if len(seq) > (max_seq_len-2):
            seq = seq[0:(max_seq_len-2)]
        seq =  ['[CLS]'] + seq + ['[SEP]']
        # ID 化
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # padding 
        padding = [0] * (max_seq_len-len(seq))
        # seq_mask
        seq_mask = [1]*len(seq) + padding
        # segment
        seg_ment = [0] * len(seq) + padding
        # 补长
        seq += padding

        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        assert len(seg_ment) == max_seq_len
        return seq,seq_mask,seg_ment


def merage_sqltable(data_sql, data_table):
    question,label_whereNum = [],[]
    agg_sql_dict = {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM"}
    for index,sql in enumerate(data_sql):
        query_text = re.sub(r'\s+','', sql['question'])
        whereNum= len(sql['sql']['conds'])
        question.append(query_text)
        label_whereNum.append(whereNum)
    assert len(question) == len(label_whereNum)

    out_features = len(set(label_whereNum))
    data_whereNum = {'query':question,'label':label_whereNum}
    data = pd.DataFrame(data_whereNum)
    return data,out_features

        
# load data
args = get_args()
train_sql,train_table,train_db,dev_sql, dev_table, dev_db, test_sql, test_table, test_db = load_dataset(args)
train_anno,out_features = merage_sqltable(train_sql, train_table)

# 产生训练数据
bert_tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=True)
processor = DataPrecessForSingleSentence(bert_tokenizer= bert_tokenizer)
seqs, seq_masks, seq_segments, labels = processor.get_input(dataset=train_anno, max_seq_len=64)
num_labels  = len(set(labels))

# 加载bert
bert_path = '/home1/lsy2018/NL2SQL/XSQL/model_dir/'
bert = BertModel.from_pretrained(bert_path,cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1)))



# 转tensor
t_seqs = torch.tensor(seqs, dtype=torch.long).to(device)
t_seq_masks = torch.tensor(seq_masks, dtype = torch.long).to(device)
t_seq_segments = torch.tensor(seq_segments, dtype = torch.long).to(device)
t_labels = torch.tensor(labels, dtype = torch.long).to(device)

train_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_labels)
train_sampler = RandomSampler(train_data)
train_dataloder = DataLoader(dataset= train_data, sampler= train_sampler,batch_size = 256)
bert.eval()

train_features = []
train_labels = []
bert.to(device)

with torch.no_grad():
    for step ,batch_data in enumerate(tqdm(train_dataloder,desc = 'Iteration')):
        batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data
        features = bert(batch_seqs, batch_seq_masks, batch_seq_segments)[1]
        train_features.append(features.detach())
        train_labels.append(batch_labels.detach())
train_features = torch.cat(train_features)
train_labels = torch.cat(train_labels)

class LogisticRegression(nn.Module):
    def __init__(self,out_features):
        super(LogisticRegression,self).__init__()
        self.classifier = nn.Sequential(*[
            nn.Linear(in_features=768, out_features=300, bias=True),            
            BatchNorm1d(300),
            ReLU(inplace=True),
            Dropout(0.5),
            nn.Linear(in_features=300, out_features=out_features, bias=True)
        ])
    def forward(self,x):
        x =self.classifier(x)
        return x
lr = LogisticRegression(out_features)
lr.to(device)
train_data = TensorDataset(train_features, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloder = DataLoader(dataset= train_data, sampler = train_sampler, batch_size = 512)
param_optimizer = list(lr.parameters())
optimizer = Adam(param_optimizer,lr=1e-04)
loss_collect = []
for i in trange(10, desc='Epoch'):
    for step, batch_data in enumerate(tqdm(train_dataloder, desc='Iteration')):    
        batch_features, batch_labels = batch_data
        print(batch_labels)
        # 对标签进行onehot编码
        # one_hot = torch.zeros(batch_labels.size(0), out_features).long()
        # one_hot.cpu()
        # one_hot_batch_labels = one_hot.scatter_(
        #     dim=1,
        #     index=torch.unsqueeze(batch_labels.cpu(), dim=1),
        #     src=torch.ones(batch_labels.size(0), out_features).long())
            
        logits = lr(batch_features)
        logits = logits.softmax(dim=1)
        print(logits)
        print(logits.size())
        print(batch_labels)
        print(batch_labels.size())
        exit()
        loss_function = CrossEntropyLoss()
        loss = loss_function(logits, batch_labels)
        loss.backward()
        loss_collect.append(loss.item())
        print("\r%f" % loss, end='')
        optimizer.step()
        optimizer.zero_grad()
