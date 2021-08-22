import os
import torch
import numpy as np;
import csv
import jieba
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn as nn

from GetWord2Vec import getw2v_100
from GetWord2Vec import getw2v_50
from GetWord2Vec import getw2v_30
from GetWord2Vec import getw2v_100_sgns_financial_bigram_char
from GetWord2Vec import getw2v_100_sgns_weibo_bigram_char

from GetWord2Vec import Tencent_AILab_ChineseEmbedding


from LoadData import load_data

# def datahelper():
#     data_excel_path = "/home/lsy2018/TextClassify/data.xlsx"
#     data = pd.read_excel(data_excel_path)
#     x = [0,2]
#     data.drop(data.columns[x],axis=1,inplace=True)
#     texts = []
#     labels_index={}
#     i = 0
#     labels_list = ['逾期业务', '交易查询', '中间业务', '优惠活动', '保险业务', '积分服务', '资料修改/查询', '卡片邮寄', '卡片管理', '渠道咨询', '高端增值服务', '还款服务', '申请审批', '分期业务', '额度办理', '账户服务', '其他业务', '开卡设密', '账单查询']
#     for each in labels_list:
#         labels_index[each] = i
#         i = i+1

#     labels = []  # list of label ids
#     for index, row in data.iterrows():
#         #print(row[1])
#         try:
#             each_transcript = eval(row[1])
#         except:
#             pass
#         speech = ''
#         for eachdict in each_transcript:
#             speech +=eachdict['speech']
#         # print(row[0],speech)
#         texts.append(speech)
#         labels.append(labels_index[row[0]])
#         #写：追加
#         # row = ['5', 'hanmeimei', '23', '81']
#         # out = open("data_pro.csv", "a", newline = "")
#         # csv_writer = csv.writer(out, dialect = "excel")
#         # csv_writer.writerow([row[0],speech])
#         with open('train.txt', 'a+') as f:
#             f.write(speech) 
        
#     return texts,labels

#datahelper()
# texts,labels = datahelper()
#texts,labels = load_data('keys_TextRank_10.csv')
#texts,labels = load_data('keys_TextRank_20.csv')
texts,labels = load_data('keys_TextRank_100_new.csv') # 0.848 60_new   

#textCNN模型
class textCNN(nn.Module):
    def __init__(self,args):
        super(textCNN, self).__init__()
        vocb_size  = args['vocb_size']
        dim = args['dim']
        n_class = args['n_class']
        max_len = args['max_len']
        embedding_matrix = args['embedding_matrix']

        #需要将预先训练好的词向量载入
        self.embedding = nn.Embedding(vocb_size,dim,_weight = embedding_matrix)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1,out_channels = 16,kernel_size = 5,stride = 1,padding =2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 32,kernel_size =5,stride = 1,padding = 2), 

            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(  # (16,64,64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        #self.out = nn.Linear(1536, n_class)
        self.out = nn.Linear(9216, n_class)
        #self.out = nn.Linear(6144, n_class)
    
    def forward(self,x):
        x = self.embedding(x)
        x=x.view(x.size(0),1,max_len,word_dim)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1) # 将（batch，outchanel,w,h）展平为（batch，outchanel*w*h）
        # print(x.shape)

        output = self.out(x)
        return output


#词表
word_vocb = []
word_vocb.append('')
for text in texts:
    for word in text:
        word_vocb.append(word)
word_vocb = set(word_vocb)
vocb_size = len(word_vocb)

#设置词表大小
nb_words = 40000
max_len = 64
word_dim = 300
n_class = 19

args = {}
if nb_words < vocb_size:
    nb_words = vocb_size

#textCNN 调用的参数
args['vocb_size']=nb_words
args['max_len']=max_len
args['n_class']=n_class
args['dim']=word_dim

EPOCH = 30

#生成相应大小的零矩阵
texts_with_id = np.zeros([len(texts),max_len])
#词表和索引的映射
word_to_idx = {word:i for i ,word in enumerate(word_vocb)}
idx_to_word={word_to_idx[word]:word for word in word_to_idx}

#每个单词对应的词向量
# embeddings_index = getw2v()
# embeddings_index = getw2v_100()
# embeddings_index = getw2v_30()
# embeddings_index = getw2v_50()
embeddings_index = getw2v_100_sgns_financial_bigram_char()
# embeddings_index = Tencent_AILab_ChineseEmbedding()


#预先处理好的词向量
embedding_matrix = np.zeros((nb_words, word_dim))
for word, i in word_to_idx.items():
    if i >= nb_words:
        continue
    if word in embeddings_index:
        embedding_vector = embeddings_index[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            # words not found in embedding index will be all-zeros.

args['embedding_matrix']=torch.Tensor(embedding_matrix)
#构建textCNN模型
# cnn=textCNN(args)
cnn=textCNN(args).cuda()

#生成训练数据，将训练数据的word 转换为word的索引
for i in range(0,len(texts)):
    if len(texts[i])<max_len:
        for j in range(0,len(texts[i])):
            texts_with_id[i][j]=word_to_idx[texts[i][j]]
        for j in range(len(texts[i]),max_len):
            texts_with_id[i][j] = word_to_idx['']
    else:
        for j in range(0,max_len):
            texts_with_id[i][j]=word_to_idx[texts[i][j]]

LR = 0.001
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
#损失函数
loss_function = nn.CrossEntropyLoss()
#训练批次大小
epoch_size=1000;
texts_len=len(texts_with_id)
print(texts_len)
#划分训练数据和测试数据
x_train, x_test, y_train, y_test = train_test_split(texts_with_id, labels, test_size=0.2, random_state=42)

# print(y_test)
# print(x_test)

test_y=torch.LongTensor(y_test)
test_x=torch.LongTensor(x_test)

train_x=x_train
train_y=y_train

test_epoch_size=300;
highest_acc = 0.5
count = 0
for epoch in range(EPOCH):

    for i in range(0,(int)(len(train_x)/epoch_size)):

        # b_x = Variable(torch.LongTensor(train_x[i*epoch_size:i*epoch_size+epoch_size]))
        b_x = Variable(torch.LongTensor(train_x[i*epoch_size:i*epoch_size+epoch_size])).cuda()

        # b_y = Variable(torch.LongTensor((train_y[i*epoch_size:i*epoch_size+epoch_size])))
        b_y = Variable(torch.LongTensor((train_y[i*epoch_size:i*epoch_size+epoch_size]))).cuda()
        output = cnn(b_x)
        loss = loss_function(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(str(i))
        print(loss)
        # print(output)
        # print(torch.topk(output, 2, largest=True, sorted=True)[1].data.squeeze())
        # print(torch.topk(output, 2, largest=True, sorted=True)[1].data.squeeze()[:,0])#最大的
        # print(torch.topk(output, 2, largest=True, sorted=True)[1].data.squeeze()[:,1])#第二大的
        #返回最大的k个元素。
        # print(torch.max(output, 1)[1].data.squeeze())
        pred_y_1 =  torch.topk(output, 2, largest=True, sorted=True)[1].data.squeeze()[:,0]
        pred_y_2 =  torch.topk(output, 2, largest=True, sorted=True)[1].data.squeeze()[:,1]
        acc1 = (b_y == pred_y_1)
        acc2 = (b_y == pred_y_2)
        acc = torch.add(acc1,acc2)
        #acc = (b_y == pred_y)
        #acc = acc.numpy().sum()    CPU
        acc = acc.cpu().numpy().sum() #GPU
        accuracy = acc / (b_y.size(0))
        #print(accuracy)


    acc_all = 0;
    for j in range(0, (int)(len(test_x) / test_epoch_size)):
        # b_x = Variable(torch.LongTensor(test_x[j * test_epoch_size:j * test_epoch_size + test_epoch_size]))
        b_x = Variable(torch.LongTensor(test_x[j * test_epoch_size:j * test_epoch_size + test_epoch_size])).cuda()
        # b_y = Variable(torch.LongTensor((test_y[j * test_epoch_size:j * test_epoch_size + test_epoch_size])))
        b_y = Variable(torch.LongTensor((test_y[j * test_epoch_size:j * test_epoch_size + test_epoch_size]))).cuda()
        test_output = cnn(b_x)
        # pred_y = torch.max(test_output, 1)[1].data.squeeze()
        # print(pred_y)
        # print(test_y)
        # acc = (pred_y == b_y)
        # acc = acc.numpy().sum()、
        pred_y_1 =  torch.topk(test_output, 2, largest=True, sorted=True)[1].data.squeeze()[:,0]
        pred_y_2 =  torch.topk(test_output, 2, largest=True, sorted=True)[1].data.squeeze()[:,1]
        acc1 = (b_y == pred_y_1)
        acc2 = (b_y == pred_y_2)
        acc = torch.add(acc1,acc2)
        #acc = (b_y == pred_y)
        #acc = acc.numpy().sum()
        acc = acc.cpu().numpy().sum() #GPU
        print("acc " + str(acc / b_y.size(0)))
        acc_all = acc_all + acc

    accuracy = acc_all / (test_y.size(0))
    print("epoch " + str(epoch) + " step " + str(i) + " " + "acc " + str(accuracy))
    if accuracy > highest_acc:
        count = 0
        highest_acc = accuracy
        torch.save(output,'model.pkl')
    else:
        count += 1
        if accuracy > 0.5 and count >= 5:
            break
print('highest_acc:', highest_acc)
