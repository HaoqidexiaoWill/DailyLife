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

logger = logging.getLogger(__name__)

def get_batch(data_dir, option, tokenizer, max_seq_length,debug = True):
    examples = []
    prev_sys = None
    num = 0
    
    if option == 'train':
        with open('{}/train.json'.format(data_dir)) as f:
            source = json.load(f)
            predicted_acts = None
    elif option == 'val':
        with open('{}/val.json'.format(data_dir)) as f:
            source = json.load(f) 
        with open('{}/BERT_dev_prediction.json'.format(data_dir)) as f:
            predicted_acts = json.load(f)
        #predicted_acts = None
    else: 
        with open('{}/test.json'.format(data_dir)) as f:
            source = json.load(f)
        with open('{}/BERT_test_prediction.json'.format(data_dir)) as f:
            predicted_acts = json.load(f)
        #predicted_acts = None

    if debug == True: source = source[:50]
    logger.info("Loading total {} dialogs".format(len(source)))
    for num_dial, dialog_info in enumerate(source):
        hist = []
        dialog_file = dialog_info['file']
        dialog = dialog_info['info']
        for turn_num, turn in enumerate(dialog):
            tokens = tokenizer.tokenize(turn['user'])
            query = copy.copy(tokens)
            if len(tokens) > max_seq_length - 2:
                query = query[:max_seq_length - 2]

            if len(hist) == 0:
                if len(tokens) > max_seq_length - 2:
                    tokens = tokens[:max_seq_length - 2]
            else:
                tokens = hist + [Constants.SEP_WORD] + tokens
                if len(tokens) > max_seq_length - 2:
                    tokens = tokens[-(max_seq_length - 2):]
            resp = [Constants.SOS_WORD] + tokenizer.tokenize(turn['sys']) + [Constants.EOS_WORD]

            if len(resp) > Constants.RESP_MAX_LEN:
                resp = resp[:Constants.RESP_MAX_LEN-1] + [Constants.EOS_WORD]
            else:
                resp = resp + [Constants.PAD_WORD] * (Constants.RESP_MAX_LEN - len(resp))
            
            resp_inp_ids = tokenizer.convert_tokens_to_ids(resp[:-1])
            resp_out_ids = tokenizer.convert_tokens_to_ids(resp[1:])

            tokens = [Constants.CLS_WORD] + tokens + [Constants.SEP_WORD]
            query = [Constants.CLS_WORD] + query + [Constants.SEP_WORD]
            
            
            input_ids = tokenizer.convert_tokens_to_ids(tokens)        

            query_ids = tokenizer.convert_tokens_to_ids(query)            
            query_padding = [Constants.PAD] * (max_seq_length - len(query_ids))
            query_ids += query_padding
            
            padding = [Constants.PAD] * (max_seq_length - len(input_ids))
            input_ids += padding
            
            assert len(input_ids) == max_seq_length
            
            act_vecs = [0] * len(Constants.act_ontology)
            if turn['act'] != "None":
                for w in turn['act']:
                    act_vecs[Constants.act_ontology.index(w)] = 1
            
            if predicted_acts is not None:
                hierarchical_act_vecs = np.asarray(predicted_acts[dialog_file][str(turn_num)], 'int64')
            else:
                hierarchical_act_vecs = np.zeros((Constants.act_len), 'int64')
                if turn['act'] != "None":
                    for w in turn['act']:
                        d, f, s = w.split('-')
                        hierarchical_act_vecs[Constants.domains.index(d)] = 1
                        hierarchical_act_vecs[len(Constants.domains) + Constants.functions.index(f)] = 1                        
                        hierarchical_act_vecs[len(Constants.domains) + len(Constants.functions) + Constants.arguments.index(s)] = 1

            
            examples.append([input_ids, resp_inp_ids, resp_out_ids, hierarchical_act_vecs, dialog_file])            
            num += 1
            if num < 5 and option == 'train': 
                logger.info("*** Example ***")
                logger.info("guid: %s" % (str(num)))
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("action_vecs: %s" % " ".join([str(x) for x in hierarchical_act_vecs]))
                logger.info("system response: %s" % " ".join([str(x) for x in resp if x != "[PAD]"]))
                logger.info("")
            
            sys = tokenizer.tokenize(turn['sys'])
            if turn_num == 0:
                hist = tokens[1:-1] + [Constants.SEP_WORD] + sys
            else:
                hist = hist + [Constants.SEP_WORD] + tokens[1:-1] + [Constants.SEP_WORD] + sys

    all_input_ids = torch.tensor([f[0] for f in examples], dtype=torch.long)
    all_response_in = torch.tensor([f[1] for f in examples], dtype=torch.long)
    all_response_out = torch.tensor([f[2] for f in examples], dtype=torch.long)
    all_hierarchical_act_vecs = torch.tensor([f[3] for f in examples], dtype=torch.float32)
    all_files = [f[4] for f in examples]


    return all_input_ids, all_response_in, all_response_out, all_hierarchical_act_vecs, all_files