from pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
import torch

tokenizer = BertTokenizer.from_pretrained('./chinese_wwm_L-12_H-768_A-12/vocab.txt')
bert = BertModel.from_pretrained('./chinese_wwm_L-12_H-768_A-12/')
bert.eval()
string1 = '没问题，那么bert返回给我们了什么呢？'
string2 = '主料已经准备好了，开始下菜，那么怎么让bert在我们的任务下跑起来呢'
string3 = '那么bert返回给我们了什么呢？'
string4 = '没问题，那么bert返回给我们了什么呢？'
string5 = '2015年港口股息率超过百分之四的港股标的以及子行业分别是什么'
string6 = '我想查询15年的港股子行业和标的，同年股息率是超过4%的'
string7 = '股息率超过4%的港口股子行业是什么，它的标的又是哪儿'

def sim(string1,string2):
    tokens1 = tokenizer.tokenize(string1)
    tokens2 = tokenizer.tokenize(string2)

    ids1 = torch.tensor([tokenizer.convert_tokens_to_ids(tokens1)])
    ids2 = torch.tensor([tokenizer.convert_tokens_to_ids(tokens2)])

    _,pool1 = bert(ids1, output_all_encoded_layers=False)
    _,pool2 = bert(ids2, output_all_encoded_layers=False)

    x = pool1.detach().numpy()
    y = pool2.detach().numpy()
    x_ = pool1.t().detach().numpy()
    y_ = pool2.t().detach().numpy()

    score = np.dot(x,y_)/(np.sqrt(np.dot(x, x_)) * np.sqrt(np.dot(y, y_)))

    return score[0][0]

'''
0.36825114
0.63507515
0.9999999
'''
# print(sim(string1,string2))
# print(sim(string1,string3))
# print(sim(string1,string4))
print(sim(string7,string6))
print(sim(string7,string5))
