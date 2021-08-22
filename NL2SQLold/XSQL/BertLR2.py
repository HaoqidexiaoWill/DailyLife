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


#pytorch_bert包提供了BertTokenizer类，从选取的模型中提取tok
from pytorch_pretrained_bert import BertTokenizer
# bert_tok = BertTokenizer.from_pretrained(config.bert_model_name,)
from Annotation import load_dataset,get_args
args = get_args()
tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=True)


#设置一个config类，便于参数配置与更改
class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


# bert 配置
config = Config(
    testing=False,
    bert_model_name="bert-base-chinese", 
    #选用中文预训练模型：Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters
    max_lr=3e-5,
    epochs=2,
    use_fp16=False, #fastai里可以方便地调整精度，加快训练速度：learner.to_fp16()
    bs=8,
    max_seq_len=128, #选取合适的seq_length，较大的值可能导致训练极慢报错等
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
    print(data['label'].unique())
    return data


# load data
train_sql,train_table,train_db,dev_sql, dev_table, dev_db, test_sql, test_table, test_db = load_dataset(args)
train_anno = merage_sqltable(train_sql, train_table)
dev_anno = merage_sqltable(dev_sql,dev_table)
# print(train_anno.shape,dev_anno.shape)
# (41522, 2) (4396, 2)


#建立TextDataBunch
databunch = TextClasDataBunch.from_df(".", train_anno, dev_anno,
                  tokenizer=fastai_tokenizer,
                  vocab=fastai_bert_vocab,
                  include_bos=False,
                  include_eos=False,
                  text_cols="query",
                  label_cols='label',
                  bs=config.bs,
                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
             )
# print(databunch.show_batch())
bert_path = '/home1/lsy2018/NL2SQL/XSQL/model_dir/'
# from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification
from model_dir.modeling import BertForQuestionAnswering, BertConfig,BertModel,BertForSequenceClassification
bert_model = BertForSequenceClassification.from_pretrained(bert_path, num_labels=4)

#二分类问题选用CrossEntrypyLoss作为损失函数
loss_func = nn.CrossEntropyLoss()


#建立Learner(数据,预训练模型,损失函数)
learner = Learner(databunch, bert_model,loss_func=loss_func,metrics=accuracy)

#开始训练
# learner.fit_one_cycle(config.epochs, max_lr=config.max_lr)
# learner.save('bertlr_0715')


# 测试
a = learner.predict('公里数在300千米以下的，又或者是其投资总额低于500亿元的高铁线路是哪到哪啊')
print(a)