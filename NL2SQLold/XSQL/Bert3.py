import os
import random
import copy
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import numpy as np
import time
import math
import gc
import re
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertAdam, BertModel
from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from apex import amp
import json
from sklearn.metrics import *



class Trainer:
    def __init__(self,data_dir,model_name,epochs = 1,batch_size = 64,base_batch_size = 32,max_len =120,part = 1,seed = 1234,debug_mode = False):
        '''
        base_batch_size
        part
        debug_mode
        '''
        self.device = torch.device('cuda')
        self.data_dir = data_dir
        self.debug_mode = debug_mode
        self.model_name = model_name
        self.seed = seed
        self.seed_everything()  
        self.max_len = max_len
        self.epochs = epochs
        self.base_batch_size = base_batch_size
        self.batch_size = batch_size
        self.split_ratio = 0.80
        if os.path.exists(self.data_dir):
            self.train_data_path = os.path.join(self.data_dir, "train/train.json")
            self.train_table_path = os.path.join(self.data_dir, "train/train.tables.json")
            self.valid_data_path = os.path.join(self.data_dir, "val/val.json")
            self.valid_table_path = os.path.join(self.data_dir, "val/val.tables.json")
            self.test_data_path = os.path.join(self.data_dir, "test/test.json")
            self.test_table_path = os.path.join(self.data_dir, "test/test.tables.json")
            # self.bert_model_path = os.path.join(self.data_dir, "chinese_L-12_H-768_A-12/")
            # self.pytorch_bert_path = os.path.join(self.data_dir, "/chinese_L-12_H-768_A-12/pytorch_model.bin")
            # self.bert_config = BertConfig(os.path.join(self.data_dir, "chinese_L-12_H-768_A-12/bert_config.json"))
        else:
            input_dir = "/home1/lsy2018/NL2SQL/XSQL/data"
            self.train_data_path = os.path.join(input_dir, "train/train.json")
            self.train_table_path = os.path.join(input_dir, "train/train.tables.json")
            self.valid_data_path = os.path.join(input_dir, "val/val.json")
            self.valid_table_path = os.path.join(input_dir, "val/val.tables.json")
            self.test_data_path = os.path.join(input_dir, "test/test.json")
            self.test_table_path = os.path.join(input_dir, "test/test.tables.json")
            # self.bert_model_path = os.path.join(kaggle_input_dir, "chinese_L-12_H-768_A-12/")
            # self.pytorch_bert_path = os.path.join(kaggle_input_dir, "/chinese_L-12_H-768_A-12/pytorch_model.bin")
            # self.bert_config = BertConfig(os.path.join(kaggle_input_dir, "chinese_L-12_H-768_A-12/bert_config.json"))  

        def load_data(self,path,num = None):
            data_list = []
            with open (path,'r') as f:
                for i ,line in enumerate(f):
                    if self.debug_mode  and i == 10:break
                    sample = json.loads(line)
                    data_list.append(sample)
            if num and not self.debug_mode:
                random.seed(self.seed)
                data_list = random.sample(data_list,num)
            print('len(data_list)',len(data_list))
            return data_list      
        def load_table(self,path):
            table_dict = {}  
            with open(path,'r') as f:
                for i , line in enumerate(f):
                    table  = json.loads(line)
                    table_dict[table['id']] = table
            return table_dict

        def seed_everything(self):
            random.seed(self.seed)
            os.environ['PYTHONHASHSEED'] = str(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            # 保证实验的可重复性
            torch.backends.cudnn.deterministic = True

        def convert_lines(self,text_serires,max_seq_length,bert_tokenizer):
            '''
            text_serires
            '''
            max_seq_length -= 2
            all_tokens = []
            for text in text_serires:
                tokens = bert_tokenizer.tokenize(text)
                if len(tokens) > max_seq_length:
                    tokens = tokens[:max_seq_length]
                one_token = bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]']) + [0]*(max_seq_length - len(tokens))
                all_tokens.append(one_token)

        def create_mask(self,max_len,start_index,mask_len):
            '''
            干嘛用的？
            似乎生成 mask_ids?
            '''
            mask = [0] *max_len
            for i in range(start_index,start_index + mask_len):
                mask[i] = 1
            return mask
        
        def process_sample(self,sample,table_dict,bert_tokenizer):
            question  = sample['question']
            table_id = sample['table_id']
            sel_list = sample['sql']['sel']
            agg_list = sample['sql']['agg']
            con_list = sample['sql']['conds']
            # 预测的contions 之间的关系
            connection = sample['sql']['conda_connn_op']
            table_title = table_dict[table_id]['title']
            tabel_header_list = table_dict[table_id]['header']
            table_row_list = table_dict[table_id]['rows']
            col_dict = {header_name: set() for header_name in tabel_header_list}
            for row in table_row_list:                    # TODO col_dict 是生成啥的
                for col,value in enumerate(row):
                    header_name = tabel_header_list[col]
                    col_dict[header_name].add(str(value)) # TODO .add函数干嘛用的
            sel_dict = {sel:agg for sel,agg in zip(sel_list,agg_list)}
            # TODO
            # <class 'list'>: [[0, 2, '大黄蜂'], [0, 2, '密室逃生']] 
            # 一列两value 多一个任务判断where一列的value数, con_dict里面的数量要和conds匹配，否则放弃这一列（但也不能作为非con非sel训练）
            # 标注只能用多分类？有可能对应多

            # TODO QuestionMatcher
            dupliciate_indices = QuestionMatcher.duplicate_relative_index(con_list)
            con_dict = {}
            # TODO
            # 为啥 duplicicate_index 是跟着value的
            for [col_col,op,value],dupliciate_index in zip(con_list,dupliciate_indices):
                value = value.strip()
                matchched_value,matched_index  =  QuestionMatcher.match_value(question, value, duplicate_index)
                if len(matchched_value) > 0:
                    if con_col in con_dict:
                        con_dict[con_col].append([op,matchched_value,matched_index])
                    else:
                        con_dict[con_col] = [[op,matchched_value,matched_index]]
                    # TODO：con_dict要看看len和conds里同一列的数量是否一致，不一致不参与训练
            # TODO：多任务加上col对应的con数量
            # TODO：需要变成训练集的就是 sel_dict、con_dict和connection
            # TODO: 只有conds的序列标注任务是valid的，其他都不valid

            # TODO conc_tokens 、 tag_mask, attention_mask,cls_index_list 是啥意思

            conc_tokens = []
            tag_masks = []
            sel_masks = []
            con_masks = []
            type_masks = []
            attention_masks = []
            header_masks = []
            question_masks = []
            value_masks = []
            connection_labels = []
            agg_labels = []
            tag_labels = []
            con_num_labels = []
            type_labels = []
            cls_index_list = []
            header_question_list = []
            header_table_id_list = []


            question_tokens = bert_tokenizer.tokenize(question)
            question_ids = bert_tokenizer.convert_tokens_to_ids(['[CLS]'+question+ ['[SEP']])
            header_cls_index = len(question_ids)
            question_mask = self.create_mask(max_len = self.max_len,start_index = 1,mask_len = len(question))
            # TODO tag_list = sample_tag_logits[j][1:cls_index-1] 为啥这句话注释掉了

            for col in range(len(tabel_header_list)):
                header = tabel_header_list[col]
                value_set = col_dict[header]
                header_tokens = bert_tokenizer.tokenize(header)
                header_ids = bert_tokenizer.convert_tokens_to_ids('[CLS]' + header_tokens + ['[SEP'])
                # TODO start_index 索引是确定的是问题数字+ 1，为啥要加这个1
                # TODO 当前的 header mask 是一个问题拼上所有的列名还是一个问题拼一个列名？
                header_mask = self.create_mask(max_len = self.max_len,start_index = len(question_ids)+1,mask_len = len(header_tokens))
                
                conc_ids = question_ids + header_ids
                value_start_index = len(conc_ids)
                for value in value_set:
                    value_tokens = bert_tokenizer.tokenize(value)
                    value_ids = bert_tokenizer.convert_tokens_to_ids(value_tokens+'[SEP]')
                    if len(conc_ids) + len(value_id) <= self.max_len:
                        conc_ids += value_ids
                value_mask_len = len(conc_ids) - value_start_index-1
                value_mask = self.create_mask(max_len=self.max_len, start_index=value_start_index, mask_len=value_mask_len)
                attention_mask = self.create_mask(max_len=self.max_len, start_index=0, mask_len=len(conc_ids))
                conc_ids = conc_ids + [0] * (self.max_len - len(conc_ids))


                # TODO  为啥4 是不标注
                tag_ids = [4]* len(conc_ids)
                sel_mask,con_mask,type_mask = 0,0,1
                connection_id,agg_id,con_num = 0,0,0

                if col in con_dict:
                    # TODO 为啥如果header 对应多个values values 必须全部匹配才能进入训练
                    # TODO map lambda 这个写法没看懂
                    header_con_list= con_dict[col]
                    if list(map(lambda x: x[0],con_list)).count(col) != len(con_dict[col]) : continue
                    for [op,value,index] in header_con_list:
                        # TODO  tag索引为啥要加1
                        # TODO　为啥　Op 作为tag id
                        tag_ids[index+1:index+1+len(value)] = [op] * len(value)
                    # TODO 为啥 len的是question question_id
                    tag_mask = [0]+ [1]*len(question) + [0]*(self.max_len-len(question)-1)
                    con_mask = 1
                    connetiocn_id = connection
                    con_num = min(len(header_con_list),3)   # 4只有一个样本，归到3类
                    type_id = 1
                # TODO sel_dict 和 con_dict 区别
                elif col in sel_dict:
                   #  TODO  是不是还有同一个 sel，col 不同的聚合方式
                   tag_mask = [0]*self.max_len
                   sel_mask = 1
                   agg_id = sel_dict[col]
                   type_id = 0 
                else:
                    tag_mask = [0] * self.mask_len
                    type_id = 2

                # TODO type_id 和 tag_mask 区别？
                conc_tokens.append(conc_ids)
                tag_masks.append(tag_mask)
                sel_masks.append(sel_mask)
                con_masks.append(con_mask)
                type_masks.append(type_mask)
                attention_masks.append(attention_mask)
                connection_labels.append(connection_id)
                agg_labels.append(agg_id)
                tag_labels.append(tag_ids)
                con_num_labels.append(con_num)
                type_labels.append(type_id)
                cls_index_list.append(header_cls_index)
                header_question_list.append(question)
                header_table_id_list.append(table_id)
                header_masks.append(header_mask)
                question_masks.append(question_mask)
                value_masks.append(value_mask)
            return tag_masks, sel_masks, con_masks, type_masks, attention_masks, connection_labels, agg_labels, tag_labels, con_num_labels, type_labels, cls_index_list, conc_tokens, header_question_list, header_table_id_list, header_masks, question_masks, value_masks


        def creadte_dataloader(self):
            '''
            sel 列 agg 类型
            where 列 逻辑符 值
            where 连接符
            
            问题开头cls：where连接符（或者新模型，所有header拼一起，预测where连接类型？）
            列的开头cls，多任务学习：1、（不选中，sel，where） 2、agg类型（0~5：agg类型，6：不属于sel） 3、逻辑符类型：（0~3：逻辑符类型，4：不属于where）
            问题部分：序列标注，（每一个字的隐层和列开头cls拼接？再拼接列所有字符的avg？），二分类，如果列是where并且是对应value的，标注为1
            '''
            # train: 41522 val: 4396 test: 4086

            train_data_list = self.load_data(self.train_data_path, num=int(41522 * self.part))
            train_table_dict = self.load_table(self.train_table_path)
            valid_data_list = self.load_data(self.valid_data_path)
            valid_table_dict = self.load_table(self.valid_table_path)
            bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)
            train_conc_tokens = []
            train_tag_masks = []
            train_sel_masks = []
            train_con_masks = []
            train_type_masks = []
            train_attention_masks = []
            train_connection_labels = []
            train_agg_labels = []
            train_tag_labels = []
            train_con_num_labels = []
            train_type_labels = []
            train_cls_index_list = []
            train_question_list = []
            train_table_id_list = []
            train_sample_index_list = []
            train_sql_list = []
            train_header_question_list = []
            train_header_table_id_list = []
            train_header_masks = []
            train_question_masks = []
            train_value_masks = []
            for sample in train_data_list:
                processed_result = self.process_sample(sample, train_table_dict, bert_tokenizer)
                train_tag_masks.extend(processed_result[0])
                train_sel_masks.extend(processed_result[1])
                train_con_masks.extend(processed_result[2])
                train_type_masks.extend(processed_result[3])
                train_attention_masks.extend(processed_result[4])
                train_connection_labels.extend(processed_result[5])
                train_agg_labels.extend(processed_result[6])
                train_tag_labels.extend(processed_result[7])
                train_con_num_labels.extend(processed_result[8])
                train_type_labels.extend(processed_result[9])
                train_cls_index_list.extend(processed_result[10])
                train_conc_tokens.extend(processed_result[11])
                train_header_question_list.extend(processed_result[12])
                train_header_table_id_list.extend(processed_result[13])
                train_header_masks.extend(processed_result[14])
                train_question_masks.extend(processed_result[15])
                train_value_masks.extend(processed_result[16])
                train_sample_index_list.append(len(train_conc_tokens))
                train_sql_list.append(sample["sql"])
                train_question_list.append(sample["question"])
                train_table_id_list.append(sample["table_id"])
            valid_conc_tokens = []
            valid_tag_masks = []
            valid_sel_masks = []
            valid_con_masks = []
            valid_type_masks = []
            valid_attention_masks = []
            valid_connection_labels = []
            valid_agg_labels = []
            valid_tag_labels = []
            valid_con_num_labels = []
            valid_type_labels = []
            valid_cls_index_list = []
            valid_question_list = []
            valid_table_id_list = []
            valid_sample_index_list = []
            valid_sql_list = []
            valid_header_question_list = []
            valid_header_table_id_list = []
            valid_header_masks = []
            valid_question_masks = []
            valid_value_masks = []
            for sample in valid_data_list:
                processed_result = self.process_sample(sample, valid_table_dict, bert_tokenizer)
                valid_tag_masks.extend(processed_result[0])
                valid_sel_masks.extend(processed_result[1])
                valid_con_masks.extend(processed_result[2])
                valid_type_masks.extend(processed_result[3])
                valid_attention_masks.extend(processed_result[4])
                valid_connection_labels.extend(processed_result[5])
                valid_agg_labels.extend(processed_result[6])
                valid_tag_labels.extend(processed_result[7])
                valid_con_num_labels.extend(processed_result[8])
                valid_type_labels.extend(processed_result[9])
                valid_cls_index_list.extend(processed_result[10])
                valid_conc_tokens.extend(processed_result[11])
                valid_header_question_list.extend(processed_result[12])
                valid_header_table_id_list.extend(processed_result[13])
                valid_header_masks.extend(processed_result[14])
                valid_question_masks.extend(processed_result[15])
                valid_value_masks.extend(processed_result[16])
                valid_sample_index_list.append(len(valid_conc_tokens))
                valid_sql_list.append(sample["sql"])
                valid_question_list.append(sample["question"])
                valid_table_id_list.append(sample["table_id"])
            train_dataset = data.TensorDataset(torch.tensor(train_conc_tokens, dtype=torch.long),
                                            torch.tensor(train_tag_masks, dtype=torch.long),
                                            torch.tensor(train_sel_masks, dtype=torch.long),
                                            torch.tensor(train_con_masks, dtype=torch.long),
                                            torch.tensor(train_type_masks, dtype=torch.long),
                                            torch.tensor(train_attention_masks, dtype=torch.long),
                                            torch.tensor(train_connection_labels, dtype=torch.long),
                                            torch.tensor(train_agg_labels, dtype=torch.long),
                                            torch.tensor(train_tag_labels, dtype=torch.long),
                                            torch.tensor(train_con_num_labels, dtype=torch.long),
                                            torch.tensor(train_type_labels, dtype=torch.long),
                                            torch.tensor(train_cls_index_list, dtype=torch.long),
                                            torch.tensor(train_header_masks, dtype=torch.long),
                                            torch.tensor(train_question_masks, dtype=torch.long),
                                            torch.tensor(train_value_masks, dtype=torch.long)
                                            )
            valid_dataset = data.TensorDataset(torch.tensor(valid_conc_tokens, dtype=torch.long),
                                            torch.tensor(valid_tag_masks, dtype=torch.long),
                                            torch.tensor(valid_sel_masks, dtype=torch.long),
                                            torch.tensor(valid_con_masks, dtype=torch.long),
                                            torch.tensor(valid_type_masks, dtype=torch.long),
                                            torch.tensor(valid_attention_masks, dtype=torch.long),
                                            torch.tensor(valid_connection_labels, dtype=torch.long),
                                            torch.tensor(valid_agg_labels, dtype=torch.long),
                                            torch.tensor(valid_tag_labels, dtype=torch.long),
                                            torch.tensor(valid_con_num_labels, dtype=torch.long),
                                            torch.tensor(valid_type_labels, dtype=torch.long),
                                            torch.tensor(valid_cls_index_list, dtype=torch.long),
                                            torch.tensor(valid_header_masks, dtype=torch.long),
                                            torch.tensor(valid_question_masks, dtype=torch.long),
                                            torch.tensor(valid_value_masks, dtype=torch.long)
                                            )
            # 将 dataset 转成 dataloader
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.base_batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.base_batch_size, shuffle=False)
            # 返回训练数据
            return train_loader, valid_loader, valid_question_list, valid_table_id_list, valid_sample_index_list, valid_sql_list, valid_table_dict, valid_header_question_list, valid_header_table_id_list

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))






if __name__ == "__main__":
    data_dir = "/root/nb/data/nl2sql_data"
    trainer = Trainer(data_dir, "model_name", epochs=30, batch_size=64, base_batch_size=64, max_len=120, part=1, debug_mode=False)
    time1 = time.time()
    trainer.train()
    print("训练时间: %d min" % int((time.time() - time1) / 60))