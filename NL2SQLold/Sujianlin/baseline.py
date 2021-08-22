import json
import re
from tqdm import tqdm
import jieba
import codecs
import numpy as np
# 句子最大长度
maxlen = 160
num_agg = 7 # agg_sql_dict = {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM", 6:"不被select"}
num_op = 5 # {0:">", 1:"<", 2:"==", 3:"!=", 4:"不被select"}
num_cond_conn_op = 3 # conn_sql_dict = {0:"", 1:"and", 2:"or"}
learning_rate = 5e-5
min_learning_rate = 1e-5

# 读取数据
def read_data(data_file, table_file):
    data, tables = [], {}
    with open(data_file) as f:
        for l in f:
            data.append(json.loads(l))
    with open(table_file) as f:
        for l in f:
            l = json.loads(l)
            d = {}
            d['headers'] = l['header']
            d['header2id'] = {j: i for i, j in enumerate(d['headers'])}
            d['content'] = {}
            d['all_values'] = set()
            rows = np.array(l['rows'])
            for i, h in enumerate(d['headers']):
                d['content'][h] = set(rows[:, i])
                d['all_values'].update(d['content'][h])
            d['all_values'] = set([i for i in d['all_values'] if hasattr(i, '__len__')])
            tables[l['id']] = d
    return data, tables
train_data, train_tables = read_data('/home1/lsy2018/NL2SQL/python3/data//train/train.json', '/home1/lsy2018/NL2SQL/python3/data/train/train.tables.json')
valid_data, valid_tables = read_data('/home1/lsy2018/NL2SQL/python3/data/val/val.json', '/home1/lsy2018/NL2SQL/python3/data/val/val.tables.json')
test_data, test_tables = read_data('/home1/lsy2018/NL2SQL/python3/data/test/test.json', '/home1/lsy2018/NL2SQL/python3/data/test/test.tables.json')
# valid_tables 格式
'''
{'43ad6bdc1d7111e988a6f40f24344a08': 
{'headers': ['300城市土地出让', '规划建筑面积(万㎡)', '成交楼面均价(元/㎡)', '平均溢价率(%)', '土地出让金(亿元)', '同比'], 
'header2id': {'300城市土地出让': 0, '规划建筑面积(万㎡)': 1, '成交楼面均价(元/㎡)': 2, '平均溢价率(%)': 3, '土地出让金(亿元)': 4, '同比': 5}, 
'content': {
    '300城市土地出让': {'2011年7月-2012年6月', '2016年7月-2017年6月', '2010年7月-2011年6月', '2018年7月-2019年6月E', '2014年7月-2015年6月', '2015年7月-2016年6月'}, 
    '规划建筑面积(万㎡)': {'150515.0', '199533.0', '189395.0', '81564.0', '244512.0', '250637.0', '198865.0', '151707.0', '168212.0', '237405.0', '155643.0'}, 
    '成交楼面均价(元/㎡)': {'1173.0', '2368.0', '910.0', '2113.0', '1037.0', '1246.0', '1040.0', '2202.0', '1260.0', '1567.0'}, 
    '平均溢价率(%)': {'None', '9.0', '15.0', '8.0', '23.0', '41.0', '6.0', '12.0', '21.0', '40.0', '28.0'}, 
    '土地出让金(亿元)': {'25750.0', '33542.0', '20056.0', '24897.0', '18479.0', '32061.0', '17261.0', '44854.0', '42020.0', '24738.0', '18885.0'}, 
    '同比': {'None', '39.0', '-41.0', '24.0', '-26.0', '36.0', '25.0', '-6.0', '34.0', '31.0'}}, 
    'all_values': {'15.0', '2009年7月-2010年6月', '81564.0', '244512.0', '2012年7月-2013年6月', '250637.0', '2017年7月-2018年6月', '18479.0', '41.0', '32061.0', '237405.0', '2016 年7月-2017年6月', '2018年7月-2018年11月', '21.0', '24738.0', '-6.0', '2014年7月-2015年6月', '25750.0', '1173.0', '199533.0', '189395.0', '33542.0', '20056.0', '39.0', '-41.0', '910.0', '2013年7月-2014年6月', '25.0', '1037.0', '155643.0', '1246.0', '1040.0', '2010年7月-2011年6月', '42020.0', '1260.0', '1567.0', '150515.0', '2011年7月-2012年6月', '8.0', '151707.0', '24897.0', '36.0', '34.0', '9.0', '6.0', '2202.0', '44854.0', '2018年7月-2019年6月E', '40.0', '2015年7月-2016年6月', 'None', '2368.0', '28.0', '24.0', '198865.0', '-26.0', '23.0', '2113.0', '168212.0', '12.0', '17261.0', '31.0', '18885.0'}}}
'''

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)


def seq_padding(X, padding=0, maxlen=None):
    if maxlen is None:
        L = [len(x) for x in X]
        ML = max(L)
    else:
        ML = maxlen
    return np.array([
        np.concatenate([x[:ML], [padding] * (ML - len(x))]) if len(x[:ML]) < ML else x for x in X
    ])

def most_similar(s, slist):
    """从词表中找最相近的词（当无法全匹配的时候）
    """
    if len(slist) == 0:
        return s
    scores = [editdistance.eval(s, t) for t in slist]
    return slist[np.argmin(scores)]

def most_similar_2(w, s):

    """从句子s中找与w最相近的片段，
    借助分词工具和ngram的方式尽量精确地确定边界。
    """
    sw = jieba.lcut(s)
    sl = list(sw)
    sl.extend([''.join(i) for i in zip(sw, sw[1:])])
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:])])
    return most_similar(w, sl)

class data_generator:
    def __init__(self,data,tables,batch_size = 32):
        self.data = data
        self.tables = tables
        self.batch_size  = batch_size
        self.steps = len(self.data)//self.batch_size
        if(len(self.data) % self.batch_size != 0):
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = range(len(self.data))
            np.random.shuffle(idxs)
            X1, X2, XM, H, HM, SEL, CONN, CSEL, COP = [], [], [], [], [], [], [], [], []
            for i in idxs:
                d = self.data[i]
                t = self.tables[d['table_id']]['headers']
                x1,x2 = tokenizer.encode(d['question'])
                xm = [0]+[1]*len(d['quesition'])+[0]
                h = []
                for j in t :
                    _x1,_x2 = tokenizer.encode(j)
                    h.append(len(x1))
                    x1.extend(_x1)
                    x2.extend(_x2)
                hm = [1]*len(h)
                sel = []
                for h in range(len(h)):
                    if j in d['sql']['sel']:
                        j = d['sql']['sel'].index 
