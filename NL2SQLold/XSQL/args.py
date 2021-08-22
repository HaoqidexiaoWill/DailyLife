import torch
import argparse
import os
import sys
seed = 42
device = torch.device("cuda", 0)
# bert_path = './chinese_wwm_L-12_H-768_A-12/'

bert_path = '/home1/lsy2018/NL2SQL/XSQL/model_dir/'



test_lines = 1689  # 多少条训练数据，即：len(features), 记得修改 !!!!!!!!!!

max_seq_length = 256
max_query_length = 16

output_dir = "./model_dir"
predict_example_files='predict.data'

max_para_num=5  # 选择几篇文档进行预测
learning_rate = 5e-5
batch_size = 16
num_train_epochs = 4
gradient_accumulation_steps = 8   # 梯度累积
num_train_optimization_steps = int(test_lines / gradient_accumulation_steps / batch_size) * num_train_epochs
log_step = int(test_lines / batch_size / 4)  # 每个epoch验证几次，默认4次

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