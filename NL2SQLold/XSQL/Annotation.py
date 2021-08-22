import json
import sys
from dbengine import DBEngine
import numpy as np
from tqdm import tqdm
import numpy
import torch

import logging
import os
import argparse
import sys
import re

from pytorch_pretrained_bert import BertTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='train.json', help='train data')
    parser.add_argument('--trainTable', type=str, default='train.tables.json', help='train table data')
    parser.add_argument('--dev', type=str, default='val.json', help='dev data')
    parser.add_argument('--devTable', type=str, default='val.tables.json', help='dev table data')
    parser.add_argument('--test', type=str, default='test.json', help='test data')
    parser.add_argument('--testTable', type=str, default='test.tables.json', help='test table data')
    parser.add_argument('--annotation', type=str, default='./annotation/', help='data annotation')
    parser.add_argument('--vocab_file', type=str, default='./chinese_wwm_L-12_H-768_A-12/vocab.txt', help='data annotation')
    args = parser.parse_args()
    args.train_data = os.path.join('./data/train/',args.train)
    args.train_dataTable = os.path.join('./data/train/',args.trainTable)
    args.dev_data = os.path.join('./data/val/',args.dev)
    args.dev_dataTable = os.path.join('./data/val/',args.devTable)
    args.test_data = os.path.join('./data/test/',args.test)
    args.test_dataTable = os.path.join('./data/test/',args.testTable)
    if not os.path.isdir(args.annotation):
        os.makedirs(args.annotation)
    return args


    
def load_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}

    for SQL_PATH in sql_paths:
        with open(SQL_PATH, encoding='utf-8') as inf:
            for idx, line in enumerate(inf):
                sql = json.loads(line.strip())
                if use_small and idx >= 1000:
                    break
                sql_data.append(sql)
        print ("Loaded %d data from %s" % (len(sql_data), SQL_PATH))

    for TABLE_PATH in table_paths:
        with open(TABLE_PATH, encoding='utf-8') as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab[u'id']] = tab
        print ("Loaded %d data from %s" % (len(table_data), TABLE_PATH))

    #使得两个数据库sql_data、table_data里的id统一
    ret_sql_data = []
    for sql in sql_data:
        if sql[u'table_id'] in table_data:
            ret_sql_data.append(sql)

    return ret_sql_data, table_data
def load_dataset(args = False):
    print ("Loading dataset")
    if args == True:
        dev_sql, dev_table = load_data(args.dev_data, args.dev_dataTable)
        dev_db = 'data/val/val.db'

        train_sql, train_table = load_data(args.train_data, args.train_dataTable)
        train_db = 'data/train/train.db'

        test_sql, test_table = load_data(args.test_data, args.test_dataTable)
        test_db = 'data/test/test.db'
    else:

        dev_sql, dev_table = load_data(os.path.join('./data/val/','val.json'), './data/val/val.tables.json')
        dev_db = 'data/val/val.db'

        train_sql, train_table = load_data('./data/train/train.json', './data/train/train.tables.json')
        train_db = 'data/train/train.db'

        test_sql, test_table = load_data('./data/test/test.json', './data/test/test.tables.json')
        test_db = 'data/test/test.db'

    return train_sql,train_table,train_db,dev_sql, dev_table, dev_db, test_sql, test_table, test_db


def string_index(String = '',subString=''):
    # String = '已知二手房上周成交量不足70万平或者当月累计成交量小于100万平，这样可以查询年初至今的累计成交量嘛'
    # subString = '70'
    start_index = String.find(subString[0])
    String_ = String[start_index:]
    end_index_  = String_.find(subString[-1])
    if end_index_ == -1:
        return start_index,-1
    else:
    # end_index = start_index + 
        return start_index,end_index_+ 1 +start_index



def merage_sqltable(data_sql, data_table):
    data_annotation = []
    for index,sql in enumerate(data_sql):
        # print(index,sql)
        query = sql['question']
        # for condition_number in range(len(sql['sql']['conds'])):
        #     print(condition_number)
        # print(len(sql['sql']['conds']))
        for index_cond,each_cond in enumerate(sql['sql']['conds']):
            # print(index,index_cond,each_cond,)
            query_text = re.sub(r'\s+','', sql['question'])
            start_index,end_index = string_index(query_text,each_cond[2])
            '''
            print(
                sql['question'], 
                data_table[sql['table_id']]['header'],
                data_table[sql['table_id']]['header'][each_cond[0]],
                each_cond,
                start_index,
                end_index
                ) 
            '''
            # print(sql['question'])
            # print(query_text)
            # print(query_text[start_index:end_index])
            # print(each_cond[2])
            if each_cond[2] == query_text[start_index:end_index]:
                data_annotation.append([
                    sql['question'],                                                        # 问题
                    data_table[sql['table_id']]['header'],                                  # 列名的集合
                    data_table[sql['table_id']]['header'][each_cond[0]],                    # 选择的列名
                    each_cond[2],                                                           # [0, 2, '长沙']                                                
                    start_index,                                                            # value 的起始索引
                    end_index])                                                             # value 的结束索引
            else:
                continue
        # if index >=500:
        #     break
    return data_annotation
def feature_numeralize_(data_anno,tokenizer,max_query_length,max_seq_length):
    features = []
    for index , data  in enumerate(data_anno):
        # print(index,data[0])
        # query_tokens = tokenizer.tokenize(data[0].strip())
        query_tokens = [char.lower() for char in re.sub(r'\s+','', data[0])]
        # print(query_tokens)
        # print(len(query_tokens),len(data[0].strip()))
        # if len(query_tokens) != len(data[0].strip()):
        #     print(data[0].strip(),len(data[0].strip()))
        #     print(query_tokens,len(query_tokens))
        #     print([char for char in data[0].strip()]),len([char for char in data[0].strip()])
        #     exit()
        # print(''.join(data[1]))
        # column_tokens = tokenizer.tokenize(''.join(data[1]).strip())
        # column_select = tokenizer.tokenize(data[2].strip())
        column_tokens = [char.lower() for char in re.sub(r'\s+','', ''.join(data[1]))]
        column_select = [char.lower() for char in re.sub(r'\s+','', data[2])]
        
        tokens = []
        segment_ids = []
        tokens.append('[CLS]')
        segment_ids.append(0)
        start_position = data[4]
        end_position = data[5]

        if data[4] is not -1:
            start_position = data[4]+1
        else:
            start_position = -1
        if data[5] is not -1:
            end_position = data[5]+1
        else:
            end_position = -1

        for each_token in query_tokens:
            tokens.append(each_token)
            segment_ids.append(0)
        tokens.append('[SEP]')
        segment_ids.append(0)

        for each_token in column_tokens:
            tokens.append(each_token)
            segment_ids.append(1)
        tokens.append('[SEP]')
        segment_ids.append(1)

        # for each_token in column_select:
        #     tokens.append(each_token)
        #     segment_ids.append(2)
        # tokens.append('[SEP]')
        # segment_ids.append(1)

        # if end_position >= max_seq_length:
        #     end_position = len(query_tokens)
        # if len(tokens) > max_seq_length:
        #     tokens[max_seq_length-1] = '[SEP]'
        #     input_ids = tokenizer.convert_tokens_to_ids(tokens[:max_seq_length])
        #     segment_ids = segment_ids[:max_seq_length]
        # else:
        #     input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(input_ids)
        assert len(input_ids) == len(segment_ids)

        features.append({
            "input_ids":input_ids,
            "input_mask":input_mask,
            "segment_ids":segment_ids,
            "start_position":start_position,
            "end_position":end_position 
            })
        # if index >=20:
        #     break
        '''
        print({
            'query':tokens,
            "input_ids":input_ids,
            "input_mask":input_mask,
            "segment_ids":segment_ids,
            "start_position":start_position,
            "end_position":end_position 
            })
        '''

    return features


def feature_numeralize(data_anno,tokenizer,max_query_length,max_seq_length):
    features = []
    for index , data  in enumerate(data_anno):

        query_tokens_ = [char.lower() for char in re.sub(r'\s+','', data[0])]
        column_tokens_ = [char.lower() for char in re.sub(r'\s+','', ''.join(data[1]))]
        column_select_ = [char.lower() for char in re.sub(r'\s+','', data[2])]
        query_tokens = tokenizer.tokenize(''.join(query_tokens_))
        column_tokens = tokenizer.tokenize(''.join(column_tokens_))
        column_select = tokenizer.tokenize(''.join(column_select_))

        tokens = []
        segment_ids = []

        tokens.append('[CLS]')
        segment_ids.append(0)
        start_position = data[4]
        end_position = data[5]
        if start_position == -1 or end_position == -1:
            continue
        else:
            start_position = start_position + 1
            end_position = end_position + 1


        if end_position > max_seq_length-1:
            continue
        if len(tokens) > max_seq_length:
            tokens[max_seq_length-1] = '[SEP]'
            input_ids = tokenizer.convert_tokens_to_ids(tokens[:max_seq_length])
            segment_ids = segment_ids[:max_seq_length]
        else:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)


        # ['二', '零', '一', '九', '年', '第', '四', '周', '大', '黄', '蜂', '和', '密', '室', '逃', '生', '这', '两', '部', '影', '片', '的', '票', '房', '总', '占', '比', '是', '多', '少', '呀']
        for each_token in query_tokens:
            tokens.append(each_token)
            segment_ids.append(0)
        tokens.append('[SEP]')
        segment_ids.append(0)

        #['影', '片', '名', '称', '周', '票', '房', '（', '万', '）', '票', '房', '占', '比', '（', '%', '）', '场', '均', '人', '次']
        for each_token in column_tokens:
            tokens.append(each_token)
            segment_ids.append(1)
        tokens.append('[SEP]')
        segment_ids.append(1)

        # # ['影', '片', '名', '称']
        # for each_token in column_select:
        #     tokens.append(each_token)
        #     segment_ids.append(2)
        # tokens.append('[SEP]')
        # segment_ids.append(2)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(input_ids)
        assert len(input_ids) == len(segment_ids)

        features.append({
            "input_ids":input_ids,
            "input_mask":input_mask,
            "segment_ids":segment_ids,
            "start_position":start_position,
            "end_position":end_position 
            })
        # if index >=20:
        #     break
        '''
        print({
            'query':tokens,
            "input_ids":input_ids,
            "input_mask":input_mask,
            "segment_ids":segment_ids,
            "start_position":start_position,
            "end_position":end_position 
            })
        '''

    return features


def feature_numeralize2(data_anno,tokenizer,max_query_length,max_seq_length):
    features = []
    for index , data  in enumerate(data_anno):

        query_tokens_ = [char.lower() for char in re.sub(r'\s+','', data[0])]
        column_select_ = [char.lower() for char in re.sub(r'\s+','', data[2])]
        query_tokens = tokenizer.tokenize(''.join(query_tokens_))
        column_select = tokenizer.tokenize(''.join(column_select_))
        
        tokens = []
        segment_ids = []

        tokens.append('[CLS]')
        segment_ids.append(0)
        start_position = data[4]
        end_position = data[5]
        if start_position == -1 or end_position == -1:
            continue
        else:
            start_position = start_position + 1
            end_position = end_position + 1

        if end_position > max_seq_length-1:
            continue



        # ['二', '零', '一', '九', '年', '第', '四', '周', '大', '黄', '蜂', '和', '密', '室', '逃', '生', '这', '两', '部', '影', '片', '的', '票', '房', '总', '占', '比', '是', '多', '少', '呀']
        for each_token in query_tokens:
            tokens.append(each_token)
            segment_ids.append(0)
        tokens.append('[SEP]')
        segment_ids.append(0)


        # ['影', '片', '名', '称']
        for each_token in column_select:
            tokens.append(each_token)
            segment_ids.append(1)
        tokens.append('[SEP]')
        segment_ids.append(1)

        if len(tokens) > max_seq_length:
            tokens[max_seq_length-1] = '[SEP]'
            input_ids = tokenizer.convert_tokens_to_ids(tokens[:max_seq_length])
            segment_ids = segment_ids[:max_seq_length]
        else:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
        

        input_mask = [1]*len(input_ids)
        assert len(input_ids) == len(segment_ids)

        features.append({
            "input_ids":input_ids,
            "input_mask":input_mask,
            "segment_ids":segment_ids,
            "start_position":start_position,
            "end_position":end_position 
            })
        # if index >=20:
        #     break
        '''
        print({
            'query':tokens,
            "input_ids":input_ids,
            "input_mask":input_mask,
            "segment_ids":segment_ids,
            "start_position":start_position,
            "end_position":end_position 
            })
        '''

    return features


def write_jsonFile(train_data,name,mode = 'feature_extration'):
    if mode == 'feature_extration':
        file_path = os.path.join(args.annotation,'{}_anno.json'.format(name))
        if os.path.exists(file_path):
            os.remove(file_path)
        fw = open(file_path,'a', encoding='utf-8')
        for index,data in enumerate(train_data):
            # print(index,data)
            json.dump({
                'query':data[0],
                'column_list':data[1],
                'column_select':data[2],
                'column_value':data[3],
                'where_value_start':data[4],
                'where_value_end':data[5]},fw,ensure_ascii=False)
            fw.writelines('\n')
    elif mode == 'numeralize':
        file_path = os.path.join(args.annotation,'{}_numera.json'.format(name))
        if os.path.exists(file_path):
            os.remove(file_path)
        with open(file_path,'a', encoding='utf-8') as fw:
            for data in tqdm(train_data):
                fw.write(json.dumps(data,ensure_ascii=False)+'\n')





def main(args):
    train_sql,train_table,train_db,dev_sql, dev_table, dev_db, test_sql, test_table, test_db = load_dataset(args)
    train_anno = merage_sqltable(train_sql, train_table)
    dev_anno = merage_sqltable(dev_sql, dev_table)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=True)
    # string = '最近在徐州举办的活动还挺多的，具体都有哪些呢'
    # print(tokenizer.tokenize(string))
    # print('gen features')
    train_features = feature_numeralize2(train_anno,tokenizer,16,64)
    dev_features = feature_numeralize2(dev_anno,tokenizer,16,64)

    # print('writing file')
    # write_jsonFile(train_anno,name = 'train')
    # write_jsonFile(dev_anno,name = 'dev')
    write_jsonFile(train_features,mode = 'numeralize',name = 'train')
    write_jsonFile(dev_features,mode = 'numeralize',name = 'dev')

    # test_anno = merage_sqltable(test_sql, test_table)
    # test_features = feature_numeralize(test_anno,tokenizer,16,256)
    # write_jsonFile(test_features,mode = 'numeralize')
    # write_jsonFile(test_anno)

if __name__ == '__main__':
    args = get_args()
    main(args)