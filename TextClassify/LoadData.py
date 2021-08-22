import os
import torch
import numpy as np;
import csv
import jieba
from sklearn.model_selection import train_test_split
import pandas as pd


def load_data(dataFile):
    data = pd.read_csv(dataFile)
    labels,texts= data.ix[:,0],data.ix[:,1] 
    labels_index={}
    i = 0
    labels_list = ['逾期业务', '交易查询', '中间业务', '优惠活动', '保险业务', '积分服务', '资料修改/查询', '卡片邮寄', '卡片管理', '渠道咨询', '高端增值服务', '还款服务', '申请审批', '分期业务', '额度办理', '账户服务', '其他业务', '开卡设密', '账单查询']
    for each in labels_list:
        labels_index[each] = i
        i = i+1

    texts_data,labels_data = [],[]
    for index in range(len(texts)):
        try:
            a = len(texts[index])
            b = len(labels[index]) 
            texts_data.append(texts[index])
            labels_data.append(labels_index[labels[index]])
        except:
            print(texts[index])
            pass

    return texts_data,labels_data

# print(load_data('keys_TextRank_10.csv'))


    # data_excel_path = "/home/lsy2018/TextClassify/data.xlsx"
    # data = pd.read_excel(data_excel_path)
    # x = [0,2]
    # data.drop(data.columns[x],axis=1,inplace=True)
    # texts = []
    # labels_index={}
    # i = 0
    # labels_list = ['逾期业务', '交易查询', '中间业务', '优惠活动', '保险业务', '积分服务', '资料修改/查询', '卡片邮寄', '卡片管理', '渠道咨询', '高端增值服务', '还款服务', '申请审批', '分期业务', '额度办理', '账户服务', '其他业务', '开卡设密', '账单查询']
    # for each in labels_list:
    #     labels_index[each] = i
    #     i = i+1

    # labels = []  # list of label ids
    # for index, row in data.iterrows():
    #     #print(row[1])
    #     try:
    #         each_transcript = eval(row[1])
    #     except:
    #         pass
    #     speech = ''
    #     for eachdict in each_transcript:
    #         speech +=eachdict['speech']
    #     # print(row[0],speech)
    #     texts.append(speech)
    #     labels.append(labels_index[row[0]])
    #     #写：追加
    #     # row = ['5', 'hanmeimei', '23', '81']
    #     # out = open("data_pro.csv", "a", newline = "")
    #     # csv_writer = csv.writer(out, dialect = "excel")
    #     # csv_writer.writerow([row[0],speech])
    #     with open('train.txt', 'a+') as f:
    #         f.write(speech) 
        

