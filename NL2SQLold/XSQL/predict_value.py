
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from model_dir.modeling import BertForQuestionAnswering, BertConfig
from Annotation import load_dataset,merage_sqltable
import torch
import args
from args import get_args
import random
import os
import json
import re
# 随机种子
random.seed(args.seed)
torch.manual_seed(args.seed)

# from dataloader import DataReader
from pytorch_pretrained_bert import BertTokenizer
args2 = get_args()
tokenizer = BertTokenizer.from_pretrained(args2.vocab_file, do_lower_case=True)
device = args.device
def get_devData():
    train_sql,train_table,train_db,dev_sql, dev_table, dev_db, test_sql, test_table, test_db = load_dataset(args2)
    dev_anno = merage_sqltable(dev_sql, dev_table)
    return dev_anno

def predict(data,model,tokenizer = tokenizer ,max_query_length = 16,max_seq_length = 64):
    acctop1,acctop2,whole = 0.0,0.0,0.0
    for index,each_data in enumerate(data):
        query_tokens_ = [char.lower() for char in re.sub(r'\s+','', each_data[0])]
        column_tokens_ = [char.lower() for char in re.sub(r'\s+','', ''.join(each_data[1]))]
        column_select_ = [char.lower() for char in re.sub(r'\s+','', each_data[2])]
        query_tokens = tokenizer.tokenize(''.join(query_tokens_))
        column_tokens = tokenizer.tokenize(''.join(column_tokens_))
        column_select = tokenizer.tokenize(''.join(column_select_))
        tokens = []
        segment_ids = []

        tokens.append('[CLS]')
        segment_ids.append(0)
        start_position = each_data[4]
        end_position = each_data[5]
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

        # print(input_ids)
        # print(segment_ids)
        # print(input_mask)
        start_prob, end_prob = model(
            torch.LongTensor(input_ids).unsqueeze(0).to(device), 
            torch.LongTensor(segment_ids).unsqueeze(0).to(device), 
            attention_mask=torch.LongTensor(input_mask).unsqueeze(0).to(device)
            ) 
        # print(start_prob)
        # print(end_prob)
        pred_start,index_start = torch.topk(start_prob,k = 2,dim = -1, largest = True, sorted = True)
        pred_end,index_end = torch.topk(end_prob,k = 2,dim = -1, largest = True, sorted = True)
        # print('start',index_start.cpu().numpy(),each_data[4]+1)
        # print('end',index_end.cpu().numpy(),each_data[5]+1)
        start_best = index_start.cpu().numpy().tolist()[0][0]
        end_best = index_end.cpu().numpy().tolist()[0][0]
        start_second = index_start.cpu().numpy().tolist()[0][1]
        end_second = index_end.cpu().numpy().tolist()[0][1]

        # print('start',index_start.cpu().numpy(),each_data[4]+1,start_best)
        # print('end',index_end.cpu().numpy(),each_data[5]+1,end_best)
        print(
            each_data[0],
            each_data[0][each_data[4]:each_data[5]],
            each_data[0][start_best-1:end_best-1]
            )

        
        if each_data[4] == start_best-1 and each_data[5] == end_best-1:
            acctop1 += 1
            whole +=1
        elif each_data[4] == start_second-1 and each_data[5] == end_second-1:
            acctop2 += 1
            whole += 1
        else:
            whole +=1
        with open('predict.txt','a+') as f:
            f.write(each_data[0]+'\t'+each_data[0][each_data[4]:each_data[5]]+'\t'+each_data[0][start_best-1:end_best-1]+'\n')

    print('acctop1',acctop1,'acctop2',acctop2,'whole',whole)
    print(float(acctop1/whole),float(acctop2/whole))
        # exit()
        



# def predict(model,data):
#     with torch.no_grad():
#         tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=True)
#         model.eval()
#         pred_answers, ref_answers = [], []
#         return 


     
if __name__ == "__main__":

    model_path = "./model_dir/best_model"
    # 准备数据

    model = BertForQuestionAnswering.from_pretrained(args.bert_path,cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1)))
    model.load_state_dict(torch.load(model_path)) #, map_location='cpu'))
    model.to(device)

    

    # predict(model)
    data_dev = get_devData()
    predict(data_dev,model)