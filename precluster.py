import pandas as pd
from collections import Counter
import pkuseg
from sklearn.model_selection import train_test_split
import codecs
from tqdm import tqdm
import logging
import json
import re
import random

import os
segDefault = pkuseg.pkuseg()
data_path = './QAPro.txt'
root_dir = os.path.dirname(__file__)
stop_words = [line.strip() for line in open('stopwords.txt', 'r').readlines()]
location_words = [line.strip() for line in open('日本地名.txt', 'r').readlines()]
location_words.remove('\ufeff日本地名的中文及英文对照表')
location_words = [re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", location) for location in location_words]

reader = pd.read_table(data_path,header = None,names=['question','answer'])
reader.drop(index=[0],inplace=True)
reader.fillna("###", inplace=True)


def generate_keywords(text):
    					            #默认分词类型
    segs = segDefault.cut(text)
    words = Counter(segs)
    
    for stopword in stop_words:
        del words[stopword]
    
    return words.most_common(200)
'''
[('日本', 12), ('地区', 10), ('关西', 9)]
'''


def data_vocabulary(data_path):

    reader = pd.read_table(data_path,header = None,names=['question','answer'])
    reader.drop(index=[0],inplace=True)
    reader.fillna("###", inplace=True)
    context = ''

    for index, row in reader.iterrows():
        context = context+row["question"]+row["answer"]

    keywords_globle_tuple = generate_keywords(context)
    keywords_globle_list = []
    for each in keywords_globle_tuple:
        keywords_globle_list.append(each[0])
    #keywords_list = [('日本', 12), ('地区', 10), ('关西', 9)]
    for eachword in keywords_globle_tuple:
        with open('./data/keywords.txt','a+') as f:
            f.write(str(eachword[0])+'\t'+str(eachword[1])+'\n')
    return keywords_globle_list,reader
'''
def add_labellist(keywords_globle_list,reader):
    data = []
    index_id =0
    for index, row in reader.iterrows():
        keywords_local_tuple = generate_keywords(row['question'])
        label_list = []
        for eachtuple in keywords_local_tuple:
            if eachtuple[0] in keywords_globle_list and len(eachtuple[0])>1:
                label_list.append(eachtuple[0])
        label = ','.join(label_list)
        data.append({'index_id':index_id,'question':row['question'],'answer':row['answer'],'label_list':label})
        index_id +=1
    data = pd.DataFrame(data)
    data.to_csv('./data/QALabel.csv')
    return data


def split_data(reader = reader):

    X_train, X_test, y_train, y_test = train_test_split(reader['question'],reader['answer'], test_size=0.2, random_state=0)
    train = pd.DataFrame({"question":X_train,"answer":y_train})
    test = pd.DataFrame({"question":X_test,"answer":y_test})
    train.to_csv("./data/train.csv", index=False,header=False)
    test.to_csv("./data/test.csv", index=False,header=False)
'''

def word_count(data_path):
    file_read = codecs.open(data_path,'r','utf-8')
    lines = file_read.readlines()
    wordcount = Counter() 
    logging.warn('Counting')
    for line in tqdm(lines):
        line = line.rstrip("\r\n").split('\t')
        context = ''.join(line)
        segs = segDefault.cut(context)
        # wordcount = wordcount + Counter(segs)
        wordcount = wordcount + Counter([x for x in segs if len(x)>1 and  u'\u4e00' <= x <= u'\u9fff'])
    logging.warn('DEL STOPWORDS')
    for stopword in stop_words:
        if stopword in wordcount:
            del wordcount[stopword]
            
    #print(wordcount_dict.most_common(20))
    # return wordcount_dict.most_common(200)
    count_tuple = wordcount.most_common(200)
    logging.warn('Writing KEYWORDS')
    with open('./data/keywords.txt','w') as f:
        for eachword in tqdm(count_tuple):
                f.write(str(eachword[0])+'\t'+str(eachword[1])+'\n')
    

    for eachword_location in location_words:
        if eachword_location in wordcount :
            del wordcount[eachword_location]
    logging.warn('Writing KEYWORDS NO LOCATION')
    with open('./data/keywords_nolocation.txt','w') as f:
        for eachword in tqdm(wordcount.most_common(200)):
                f.write(str(eachword[0])+'\t'+str(eachword[1])+'\n')
    
    return count_tuple
            
def add_labellist(keywords_globle_tuple):
    keywords_globle_list = []
    for each in keywords_globle_tuple:
        keywords_globle_list.append(each[0])
    data_dict = {}
    file_read = codecs.open(data_path,'r','utf-8')
    lines = file_read.readlines()
    index = 0
    logging.warn('Adding Label')
    for line in tqdm(lines):
        line = line.rstrip("\r\n").split('\t')
        if len(line) <2:
            continue      
        keywords_local_tuple = generate_keywords(line[0])

        label_list = []
        for eachtuple in keywords_local_tuple:
            if eachtuple[0] in keywords_globle_list:
                #print(eachtuple[0])
                label_list.append(eachtuple[0])
        label = ','.join(label_list)
        #print(label)
        query_answer = {'question':line[0],'answer':line[1],'label_list':label}
        data_dict[index] = query_answer
        index += 1
    with open("./data/QALabel.json",'w',encoding='utf-8') as json_file:
        json.dump(data_dict,json_file,ensure_ascii=False)

def split_data(data_path):

    #name = 'QALabel'
    name = 'QQsimilarity'

    with open(data_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    #print(len(data))
    data_train = {}
    data_test = {}
    sample_index=random.sample(range(0,len(data)),round(len(data)*0.02))

    index = 0
    for key,value in tqdm(data.items()):
        #print(sample_index)
        if index in sample_index:
            data_test[key] = value 
        else:
            data_train[key] = value
        index +=1
    logging.warn('Writing TO FILE')
    with open("./data/{}_train.json".format(name),'w',encoding='utf-8') as json_file:
        json.dump(data_train,json_file,ensure_ascii=False)
    with open("./data/{}_test.json".format(name),'w',encoding='utf-8') as json_file:
        json.dump(data_test,json_file,ensure_ascii=False)


    


if __name__ == '__main__':
    # data_path = './QA.txt'

    data = os.path.join(root_dir, 'data')
    if not os.path.isdir(data):
        os.makedirs(data)
    # data_path = './QAPro.txt'

    #keywords_globle_tuple = word_count(data_path)
    #keywords = [line.strip() for line in open('./data/keywords.txt', 'r').readlines()]
    # add_labellist(keywords_globle_tuple)
    #add_labellist(keywords)
    #split_data('./data/QALabel.json')
    # split_data('./data/QALabel_demo.json')
    split_data('./data/dataToJsonBig.json')


