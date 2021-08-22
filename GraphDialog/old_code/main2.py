from config import parse_opt
import json
import torch
import random
import numpy
import logging
import os
import sys
import time
import copy
from torch.autograd import Variable
# from transformer.Transformer import Transformer, TransformerDecoder, TableSemanticDecoder
from transformer.Transformer import TableSemanticDecoder
from torch.optim.lr_scheduler import MultiStepLR
import transformer.Constants as Constants
from itertools import chain
from MultiWOZ_ import get_batch
from transformer.LSTM import LSTMDecoder
from transformer import Constants
from transformer.Semantic_LSTM import SCLSTM
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tools import *
from collections import OrderedDict
from evaluator import evaluateModel

from Utils.utils import get_device,get_train_logger
from DataSet.MultiWoz import MultiWOZDataset
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
args = parse_opt()
print(1)



class Trainer:
    def __init__(self):
        self.logger = get_train_logger('log2.txt')
        self.device =  torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.outfile = "/tmp/results.txt.pred.{}".format(args.model)
        with open("{}/vocab.json".format(args.data_dir), 'r') as f:
            self.vocabulary = json.load(f)
        self.act_ontology = Constants.act_ontology
        self.vocab, self.ivocab = self.vocabulary['vocab'], self.vocabulary['rev']
        self.tokenizer = Tokenizer(self.vocab, self.ivocab, False)
        os.makedirs(args.output_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(args.output_dir, args.model)
        self.BLEU_calc = BLEUScorer()
        self.F1_calc = F1Scorer()
        self.batch_size = args.batch_size
        self.debug = False

        self.early_stop = 50

        self.dialogs = json.load(open('{}/val.json'.format(args.data_dir)))
        self.gt_turns = json.load(open('{}/val_reference.json'.format(args.data_dir)))

        self.train_data_path = os.path.join(args.data_dir,'train.json')
        self.val_data_path = os.path.join(args.data_dir,'val.json')
        self.test_data_path = os.path.join(args.data_dir,'test.json')





    def train(self):
        
        dataset = MultiWOZDataset(
            data_dir =args.data_dir,
            logger = self.logger,
            tokenizer = self.tokenizer,
            max_seq_length =  args.max_seq_length)
        *train_examples, _  = dataset.load_data('train')
        # *train_examples, _ = get_batch(args.data_dir, 'train', self.tokenizer, args.max_seq_length,debug = self.debug)
        train_data = TensorDataset(*train_examples)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)
        # *val_examples, val_id = get_batch(args.data_dir, 'val', self.tokenizer, args.max_seq_length)
        *val_examples,val_id = dataset.load_data('val')

        eval_data = TensorDataset(*val_examples)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)


        decoder = TableSemanticDecoder(
            vocab_size=self.tokenizer.vocab_len, 
            d_word_vec=args.emb_dim, n_layers=args.layer_num,
            d_model=args.emb_dim, n_head=args.head, dropout=args.dropout)

        decoder.to(self.device)
        loss_func = torch.nn.BCELoss()
        loss_func.to(self.device)

        ce_loss_func = torch.nn.CrossEntropyLoss(ignore_index=Constants.PAD)
        ce_loss_func.to(self.device)

        decoder.train()
        self.logger.info("Start Training with {} batches".format(len(train_dataloader)))
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, decoder.parameters()), betas=(0.9, 0.98), eps=1e-09)
        scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150, 200], gamma=0.5)
        best_BLEU = 0
        early_stop_times = 0
        for epoch in range(360):
            if early_stop_times >= self.early_stop: break
            for step,batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, act_vecs, query_results, rep_in, resp_out, belief_state, hierachical_act_vecs, *_ = batch

                decoder.zero_grad()
                optimizer.zero_grad()

                logits = decoder(tgt_seq=rep_in, src_seq=input_ids, act_vecs=hierachical_act_vecs)

                loss = ce_loss_func(logits.contiguous().view(logits.size(0) * logits.size(1), -1).contiguous(),resp_out.contiguous().view(-1))
                
                loss.backward()
                optimizer.step()
                if step % 100 == 0: print("epoch {} step {} training loss {}".format(epoch, step, loss.item()))       

            scheduler.step()      

            # if loss.item() < 3.0 and epoch > 0 and epoch % args.evaluate_every == 0:
            if loss.item() < 3.0 and epoch > 0:
                
                self.logger.info("start evaluating BLEU on validation set")
                decoder.eval()
                model_turns = {}
                TP, TN, FN, FP = 0, 0, 0, 0            
                for batch_step, batch in enumerate(eval_dataloader):
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, segment_ids, act_vecs, query_results,rep_in, resp_out, belief_state, pred_hierachical_act_vecs, *_ = batch

                    hyps = decoder.translate_batch(
                        act_vecs=pred_hierachical_act_vecs, 
                        src_seq=input_ids, n_bm=args.beam_size, 
                        max_token_seq_len=40)
                    
                    for hyp_step, hyp in enumerate(hyps):
                        pred = self.tokenizer.convert_id_to_tokens(hyp)
                        file_name = val_id[batch_step * args.batch_size + hyp_step]
                        if file_name not in model_turns: model_turns[file_name] = [pred]
                        else: model_turns[file_name].append(pred)
                BLEU = self.BLEU_calc.score(model_turns, self.gt_turns)
        
                self.logger.info("{} epoch, Validation BLEU {} ".format(epoch, BLEU))
                if BLEU>28:
                    with open('predict.txt','a+') as f:
                        for i,(each_pre,each_gt) in enumerate(zip(model_turns,self.gt_turns)):
                            f.write(str(each_pre)+'\t'+str(each_gt)+'\n')
                exit()

                if BLEU > best_BLEU:
                    torch.save(decoder.state_dict(), self.checkpoint_file)
                    best_BLEU = BLEU
                    early_stop_times = 0
                else:
                    early_stop_times += 1

                decoder.train()
   

def main():
    trainer = Trainer()
    trainer.train()
if __name__ == "__main__":
    
    main()