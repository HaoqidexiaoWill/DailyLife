# coding=utf-8
import random
import numpy as np
import time
from tqdm import tqdm
import os

import torch
import torch.nn as nn


from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam

from Utils.utils import get_device,get_train_logger
from Utils.load_datatsets import load_data

from train_evalute import train, evaluate, evaluate_save
import args

class Trainer:
    def __init__(self,config,data_dir,bert_vocab_file, bert_model_dir,model_name,label_list,data_name,output_dir,log_dir):
        
        self.seed = config.seed
        self.data_dir = data_dir
        self.max_seq_length = config.max_seq_length
        self.label_list = label_list
        self.model_name = model_name
        self.bert_vocab_file = bert_vocab_file
        self.bert_model_dir = bert_model_dir
        self.output_dir = os.path.join(output_dir,data_name,model_name)
        
        # 输出的模型文件

        if not os.path.exists(self.output_dir):os.makedirs(self.output_dir)

        self.output_model_file = os.path.join(self.output_dir,WEIGHTS_NAME)  
        self.output_config_file = os.path.join(self.output_dir,CONFIG_NAME)
        self.logger = get_train_logger(os.path.join(self.output_dir,'log.txt'))

        # GPU
        self.gpu_ids = [int(device_id) for device_id in config.gpu_ids.split()]
        self.device = get_device(self.gpu_ids[0])  
        self.train_batch_size = config.train_batch_size // config.gradient_accumulation_steps
        self.dev_batch_size = config.dev_batch_size 
        self.test_batch_size = config.test_batch_size
        self.seed_everything()
        # # 分词器选择
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file)  
        self.num_labels = len(label_list)

    def seed_everything(self):
        # 设定随机种子 
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
    def get_model(self):
        if self.model_name == "BertOrigin":
            from BertOrigin.BertOrigin import BertOrigin
            model = BertOrigin.from_pretrained(self.bert_model_dir, num_labels=self.num_labels)

        return model.to(self.device)

    def train_(self):
        train_dataloader, train_examples_len = load_data(self.data_dir, self.tokenizer, self.max_seq_length, self.train_batch_size, "train",self.label_list)
        dev_dataloader, _ = load_data(self.data_dir, self.tokenizer, self.max_seq_length, self.dev_batch_size, "dev", self.label_list)
        num_train_optimization_steps = int(train_examples_len / self.train_batch_size / config.gradient_accumulation_steps) * config.num_train_epochs
        model = self.get_model()

        """ 优化器准备 """
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=config.learning_rate,
                             warmup=config.warmup_proportion,
                             t_total=num_train_optimization_steps)

        """ 损失函数准备 """
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(self.device)

        train(
            config.num_train_epochs, 
            model, train_dataloader, dev_dataloader, optimizer,criterion, config.gradient_accumulation_steps, 
            self.device, self.label_list, self.output_model_file, self.output_config_file, 
            self.logger, config.print_step, config.early_stop)
    def test_(self):
        # test 数据
        test_dataloader, _ = load_data(self.data_dir, self.tokenizer, config.max_seq_length, config.test_batch_size, "test", self.label_list)
        # 加载模型 
        bert_config = BertConfig(self.output_config_file)
        if self.model_name == "BertOrigin":
            from BertOrigin.BertOrigin import BertOrigin
            model = BertOrigin(bert_config, num_labels=self.num_labels)
        model.load_state_dict(torch.load(self.output_model_file))
        model.to(self.device)
        
        # 损失函数准备
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(self.device)

        # test the model
        test_loss, test_acc, test_report, test_auc, all_idx, all_labels, all_preds = evaluate_save(
            model, test_dataloader, criterion, self.device, self.label_list)

        #保存预测结果    
        with open(self.model_name+'/pred.txt','w') as f:
            for pred in all_preds:
                f.write(str(pred)+'\n')

        self.logger.info("-------------- Test -------------")
        self.logger.info(f'\t  Loss: {test_loss: .3f} | Acc: {test_acc*100: .3f} % | AUC:{test_auc}')

        for label in self.label_list:
            self.logger.info('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
        self.logger.info_list = ['macro avg', 'weighted avg']

        for label in self.logger.info_list:
            self.logger.info('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
def main(config):

    
    model_name = "BertOrigin"
    label_list = ['0', '1','2']
    data_name = '情感分析'
    data_dir = "ccf_data"
    output_dir = "model_save/" 
    log_dir = "log/" 

    # bert-base
    bert_vocab_file = "bert-base-chinese-vocab.txt"
    bert_model_dir = "bert-base-chinese/"

    
    trainer = Trainer(config,data_dir,bert_vocab_file, bert_model_dir,model_name,label_list,data_name,output_dir,log_dir)
    trainer.train_()
    trainer.test_()

if __name__ == "__main__":
    config = args.get_args()
    main(config)
