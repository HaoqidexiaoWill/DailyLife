#引入必要的依赖包
import numpy as np
import pandas as pd

from pathlib import Path
from typing import *

import torch
import torch.optim as optim

from fastai import *
from fastai.vision import *
from fastai.text import *
from fastai.callbacks import *
from pytorch_pretrained_bert import BertTokenizer
import json
bert_model_path = "/home1/lsy2018/BERT文本相似度/uncased_L-12_H-768_A-12/"
pytorch_bert_path =  "/home1/lsy2018/BERT文本相似度/uncased_L-12_H-768_A-12/pytorch_model.bin"
# bert_config = BertConfig("/home1/lsy2018/BERT文本相似度/uncased_L-12_H-768_A-12/bert_config.json")

from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification
#分割训练集train与验证集val
from sklearn.model_selection import train_test_split
tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case=True)
#设置一个config类，便于参数配置与更改


#设置一个config类，便于参数配置与更改
class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

config = Config(
    testing=False,
    bert_model_name="bert-base-chinese", 
    #选用中文预训练模型：Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters
    max_lr=3e-5,
    epochs=7,
    use_fp16=False, #fastai里可以方便地调整精度，加快训练速度：learner.to_fp16()
    bs=8,
    max_seq_len=512, #选取合适的seq_length，较大的值可能导致训练极慢报错等
)

class FastAiBertTokenizer(BaseTokenizer):
    """Wrapper around a BertTokenizer to be a BaseTokenizer in fastai"""
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length"""
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]
#最终的tokenizer
fastai_tokenizer = Tokenizer(
    tok_func=FastAiBertTokenizer(tokenizer, max_seq_len=config.max_seq_len), 
    pre_rules=[], 
    post_rules=[]
)

#check the vocab
print(list(tokenizer.vocab.items())[1090:1110])
#设置vocab
fastai_bert_vocab = Vocab(list(tokenizer.vocab.keys()))


def load_data(data_path = '/home1/lsy2018/RNN文本分类/data/train.json'):
    data,text,label = [],[],[]
    with open (data_path,encoding = 'utf-8') as inf:
        for index ,line in enumerate(inf):
            # if index == 50 :break
            each_data = json.loads(line.strip())
            data.append(each_data)
    for i,each_line in enumerate(data):
        file_name = each_line['id']
        turn = each_line['id']
        text_additional = ''
        text_title = each_line['title']
        text_context = each_line['context']
        text.append(text_title+text_context)
        if each_line['label'] ==  [1,0,0]:label.append('0')
        elif each_line['label'] ==  [0,1,0]:label.append('1')
        else:label.append('2')
        
    print(len(text))
    print(len(label))
    assert len(text) == len(label)
    out_features = len(set(label))
    each_data = {'query':text,'label':label}
    data_final = pd.DataFrame(each_data)
    print(data_final['label'].unique())
    return data_final
# load data
train_anno = load_data()

train_anno, val_anno = train_test_split(train_anno)
train_anno.to_csv('./data/train.csv', index = False, header = False)
val_anno.to_csv('./data/dev.csv',index = False,header = False)
print(train_anno.shape,val_anno.shape)
def get_metrics():
    return [accuracy,AUROC(),error_rate]
databunch = TextClasDataBunch.from_df("./", train_anno, val_anno,
                  tokenizer=fastai_tokenizer,
                  vocab=fastai_bert_vocab,
                  include_bos=False,
                  include_eos=False,
                  text_cols="query",
                  label_cols='label',
                  bs=config.bs,
                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
             )
bert_path = '/home1/lsy2018/NL2SQL/XSQL/model_dir/'
bert_model = BertForSequenceClassification.from_pretrained(bert_model_path, num_labels=3)
loss_func = nn.CrossEntropyLoss()
learner = Learner(databunch, bert_model,loss_func=loss_func,metrics=get_metrics())
learner.fit_one_cycle(config.epochs, max_lr=config.max_lr)

learner.export("fastai_model/bertlr_0715.pkl")
learner = load_learner('./fastai_model','bertlr_0715.pkl')


test_path = '/home1/lsy2018/RNN文本分类/data/test.json'
with open (data_path,encoding = 'utf-8') as inf:
    data,id,text = [],[],[]
    for index ,line in enumerate(inf):
        # if index == 50 :break

        each_data_ = json.loads(line.strip())

        each_data = {'id':each_data['id'],'text':each_data['title']+each_data['context']}

        data.append(each_data['title']+each_data['context'])

preds = [str(learner.predict(x)) for x in data.text]