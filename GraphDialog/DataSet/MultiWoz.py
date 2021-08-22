from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import time
import json
import numpy as np
import torch
import logging
from transformer import Constants
import copy

class MultiWOZDataset:
    def __init__(self,data_dir,logger,tokenizer,max_seq_length):
        self.data_dir = data_dir
        self.train_data_path = os.path.join(self.data_dir,'train.json')
        self.val_data_path = os.path.join(self.data_dir,'val.json')
        self.test_data_path = os.path.join(self.data_dir,'test.json')
        self.debug = False
        self.logger = logger
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.predicted_acts = None

        with open (self.train_data_path) as f:
            self.train_data  = json.load(f)
            
        with open (self.val_data_path) as f:
            self.val_data = json.load(f)
        with open (self.test_data_path) as f:
            self.test_data = json.load(f)        

    def load_data(self,mode = 'train'):
        if mode == 'train': data = self.train_data
        elif mode == 'val': data = self.val_data
        elif mode == 'test':data = self.test_data
        else: raise RuntimeError('没这个数据集')
        if self.debug == True:  data = data[:50]

        self.logger.info("Loading total {} dialogs".format(len(data)))
        examples = []
        num = 0

        for dialog_index,dialog_info in enumerate(data):
            hist = []
            hist_segment = []
            dialog_file = dialog_info['file']
            each_dialog = dialog_info['info']
            for turn_index,each_turn in enumerate(each_dialog):
                tokens = self.tokenizer.tokenize(each_turn['user'])
                query = copy.copy(tokens)
                if len(tokens) > self.max_seq_length - 2:
                    query = query[:self.max_seq_length - 2]
                segment_user = 1
                segment_sys = 2
                if len(hist) == 0:
                    if len(tokens) > self.max_seq_length - 2:
                        tokens = tokens[:self.max_seq_length - 2]
                    segment_ids = [segment_user] * len(tokens)
                else:
                    segment_ids = hist_segment + [Constants.PAD] + [segment_user] * len(tokens)
                    tokens = hist + [Constants.SEP_WORD] + tokens
                    if len(tokens) > self.max_seq_length - 2:
                        tokens = tokens[-(self.max_seq_length - 2):]
                        segment_ids = segment_ids[-(self.max_seq_length - 2):]
                resp = [Constants.SOS_WORD] + self.tokenizer.tokenize(each_turn['sys']) + [Constants.EOS_WORD]

                if len(resp) > Constants.RESP_MAX_LEN:
                    resp = resp[:Constants.RESP_MAX_LEN-1] + [Constants.EOS_WORD]
                else:
                    resp = resp + [Constants.PAD_WORD] * (Constants.RESP_MAX_LEN - len(resp))
                

                # resp_inp_ids = self.tokenizer.convert_tokens_to_ids(resp[:-1])
                # resp_out_ids = self.tokenizer.convert_tokens_to_ids(resp[1:])
                resp_inp_ids = self.tokenizer.convert_tokens_to_ids(resp)
                resp_out_ids = self.tokenizer.convert_tokens_to_ids(resp)

                bs = [0] * len(Constants.belief_state)
                if each_turn['BS'] != "None":
                    for domain in each_turn['BS']:
                        for key, value in each_turn['BS'][domain]:
                            bs[Constants.belief_state.index(domain + '-' + key)] = 1
                
                if each_turn['KB'] == 0:
                    query_results = [1, 0, 0, 0]
                elif each_turn['KB'] == 2:
                    query_results = [0, 1, 0, 0]
                elif each_turn['KB'] == 3:
                    query_results = [0, 0, 1, 0]
                elif each_turn['KB'] >= 4:
                    query_results = [0, 0, 0, 1]

                tokens = [Constants.CLS_WORD] + tokens + [Constants.SEP_WORD]
                query = [Constants.CLS_WORD] + query + [Constants.SEP_WORD]
                
                segment_ids = [Constants.PAD] + segment_ids + [Constants.PAD]
                
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)        
                input_mask = [1] * len(input_ids)

                query_ids = self.tokenizer.convert_tokens_to_ids(query)            
                query_mask = [1] * len(query_ids)
                query_segment_ids = [1] * len(query_mask)
                
                query_padding = [Constants.PAD] * (self.max_seq_length - len(query_ids))
                query_ids += query_padding
                query_mask += query_padding
                padded_query_segment_ids = query_segment_ids + query_padding
                
                padding = [Constants.PAD] * (self.max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                padded_segment_ids = segment_ids + padding
                
                assert len(input_ids) == len(input_mask) == len(padded_segment_ids) == self.max_seq_length
                
                act_vecs = [0] * len(Constants.act_ontology)
                if each_turn['act'] != "None":
                    for w in each_turn['act']:
                        act_vecs[Constants.act_ontology.index(w)] = 1
                
                hierarchical_act_vecs = np.zeros((Constants.act_len), 'int64')
                if each_turn['act'] != "None":
                    for w in each_turn['act']:
                        d, f, s = w.split('-')
                        hierarchical_act_vecs[Constants.domains.index(d)] = 1
                        hierarchical_act_vecs[len(Constants.domains) + Constants.functions.index(f)] = 1                        
                        hierarchical_act_vecs[len(Constants.domains) + len(Constants.functions) + Constants.arguments.index(s)] = 1


                examples.append([input_ids, input_mask, padded_segment_ids, act_vecs, \
                                query_results, resp_inp_ids, resp_out_ids, bs, hierarchical_act_vecs, dialog_file])            
                num += 1
                if num < 5 and mode == 'train': 
                    self.logger.info("*** Example ***")
                    self.logger.info("guid: %s" % (str(num)))
                    self.logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                    self.logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    self.logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    self.logger.info("segment_ids: %s" % " ".join([str(x) for x in padded_segment_ids]))
                    self.logger.info("action_vecs: %s" % " ".join([str(x) for x in hierarchical_act_vecs]))
                    self.logger.info("query results: %s" % " ".join([str(x) for x in query_results]))
                    self.logger.info("belief states: %s" % " ".join([str(x) for x in bs]))
                    self.logger.info("system response: %s" % " ".join([str(x) for x in resp if x != "[PAD]"]))
                    self.logger.info("")
                
                sys = self.tokenizer.tokenize(each_turn['sys'])
                if turn_index == 0:
                    hist = tokens[1:-1] + [Constants.SEP_WORD] + sys
                    hist_segment = segment_ids[1:-1] + [Constants.PAD] + [segment_sys] * len(sys)
                else:
                    hist = hist + [Constants.SEP_WORD] + tokens[1:-1] + [Constants.SEP_WORD] + sys
                    hist_segment = hist_segment + [Constants.PAD] + segment_ids[1:-1] + [Constants.PAD] + [segment_sys] * len(sys) 

        all_input_ids = torch.tensor([f[0] for f in examples], dtype=torch.long)
        all_input_mask = torch.tensor([f[1] for f in examples], dtype=torch.long)
        all_segment_ids = torch.tensor([f[2] for f in examples], dtype=torch.long)
        all_act_vecs = torch.tensor([f[3] for f in examples], dtype=torch.float32)
        all_query_results = torch.tensor([f[4] for f in examples], dtype=torch.float32)
        all_response_in = torch.tensor([f[5] for f in examples], dtype=torch.long)
        all_response_out = torch.tensor([f[6] for f in examples], dtype=torch.long)
        all_belief_state = torch.tensor([f[7] for f in examples], dtype=torch.float32)   
        all_hierarchical_act_vecs = torch.tensor([f[8] for f in examples], dtype=torch.float32)
        all_files = [f[9] for f in examples]

        return all_input_ids, all_input_mask, all_segment_ids, all_act_vecs, all_query_results, all_response_in, all_response_out, all_belief_state,all_hierarchical_act_vecs, all_files