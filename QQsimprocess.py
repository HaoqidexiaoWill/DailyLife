import pandas as pd
import Levenshtein
import numpy as np

data_path = '../data/Docomo/QuestionsPro.txt'

reader = pd.read_table(data_path, low_memory=False,header = None,names=['id','question1','question2'])
reader.fillna("###", inplace=True)

context1 = '日本旅游签证多少钱'
context2 = '代办日本个人旅游签证多少钱'
def similarity(context1,context2):
    # print(Levenshtein.ratio(context1, context2))
    # print( Levenshtein.distance(context1, context2))
    if float(Levenshtein.ratio(context1, context2)) > float(0.5):
        return 1
    else:
        return 0

# similarity(context1,context2)

reader['sim'] = reader.apply(lambda x: similarity(x.question1, x.question2), axis = 1)

#检查结果
'''
print(reader.head()) 
print(reader.query('sim==1'))
count_1 = np.sum(list(map(lambda x: x >0, reader['sim'])))
print('相似的问题数目为:{}'.format(count_1))
'''

#分割数据集
train_data = reader.sample(frac = 0.8,random_state = 0,axis = 0)
valid_test_data = reader[~reader.index.isin(train_data.index)]
valid_data = valid_test_data.sample(frac = 0.5,random_state = 0,axis = 0)
test_data = valid_test_data[~valid_test_data.index.isin(valid_data.index)]
# print(test_data.head())

#写入文件
train_data.to_csv("../data/Docomo/train_data_bimpm.csv", index=False,header=False)
valid_data.to_csv("../data/Docomo/valid_data_bimpm.csv", index=False,header=False)
test_data.to_csv("../data/Docomo/test_data_bimpm.csv", index=False,header=False)
