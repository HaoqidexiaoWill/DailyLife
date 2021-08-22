import json 
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import pandas as pd
import re
import unicodedata
import itertools
import random
import config

class GreedySearchDecoder(nn.Module):
    def __init__(self, model):
        super(GreedySearchDecoder, self).__init__()
        self.model = model

    def forward(self, data_batch):
        # Forward input and section through encoder model
        sec_hidden,encoder_outputs, encoder_hidden = self.model.encode(data_batch)#(L,B,H) (layer*direc,B,H)
        max_input_len, beam_size, hidden_size = encoder_outputs.shape

        _,_,_,_,batch_oovs,max_oov_length,extend_inp,target_variable, mask, max_target_len,extend_output = data_batch
        
        extend_inp = extend_inp.to(config.device)
        
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[-config.decoder_num_layers:]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, beam_size, device=config.device, dtype=torch.long) * config.SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=config.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=config.device)
        # Iteratively decode one word token at a time
        for _ in range(config.MAX_LENGTH):
            # Forward pass through decoder 
            decoder_output, decoder_hidden = self.model.decoder(decoder_input, sec_hidden, decoder_hidden, encoder_outputs,max_oov_length,extend_inp)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            decoder_input[decoder_input >= self.model.vocab_size] = config.UNK_token
            decoder_scores = decoder_scores.unsqueeze(0)#(1,B)
            decoder_input = decoder_input.unsqueeze(0) #(1,B)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
        # Return collections of word tokens and scores
        all_tokens = all_tokens.transpose(0,1).tolist()
        s = all_scores.gt(0).sum(dim=0,dtype=torch.float)
        all_scores = torch.div(all_scores.sum(dim=0),s)
        return all_tokens, all_scores.tolist(),batch_oovs #(B,L), (B), (B,x)

class Beam(object):
    def __init__(self, tokens, log_probs, sec_hidden, decoder_hidden, encoder_outputs):
        self.tokens = tokens
        self.log_probs = log_probs
        self.sec_hidden = sec_hidden
        self.decoder_hidden = decoder_hidden
        self.encoder_outputs = encoder_outputs
        
    def extend(self, token, log_prob, sec_hidden, decoder_hidden, encoder_outputs):
        return Beam(tokens = self.tokens+[token], 
                   log_probs = self.log_probs+[log_prob],
                   sec_hidden = sec_hidden,
                   decoder_hidden = decoder_hidden,
                   encoder_outputs = encoder_outputs)
    
    @property
    def latest_token(self):
        return self.tokens[-1]
    
    @property
    def avg_log_prob(self):
        return sum(self.log_probs)/len(self.tokens)

class BeamSearchDecoder(nn.Module):
    def __init__(self, model):
        super(BeamSearchDecoder, self).__init__()
        self.model = model
    
    def sort_beams(self,beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def forward(self, data_batch):
        # Forward input and section through encoder model
        sec_hidden,encoder_outputs, encoder_hidden = self.model.encode(data_batch)#(L,B,H) (layer*direc,B,H) (layer*direc,B,H)
        max_input_len,beam_size,hidden_size = encoder_hidden.size()
        _,_,_,_,batch_oovs,max_oov_length,extend_inp,target_variable, mask, max_target_len,extend_output = data_batch
        
        extend_inp = extend_inp.to(config.device)
        
        all_tokens = []
        all_scores = []

        for i in range(beam_size):
            tokens, score = self.beam_search(encoder_hidden[:,i].unsqueeze(1),sec_hidden[:,i].unsqueeze(1),
                encoder_outputs[:,i].unsqueeze(1),max_oov_length,extend_inp[:,i].unsqueeze(1))
            # Record token and score
            all_tokens.append(tokens)
            all_scores.append(score)
        return all_tokens,all_scores,batch_oovs #(B,L) (B,), (B,x)


    def beam_search(self, encoder_hidden, sec_hidden, encoder_outputs, max_oov_length, extend_inp):
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[-config.decoder_num_layers:]
        beams = [Beam(tokens=[config.SOS_token],
                     log_probs=[0.0],
                     sec_hidden=sec_hidden,
                     decoder_hidden=decoder_hidden,
                     encoder_outputs=encoder_outputs) for _ in range(config.beam_size)]
        
        all_sec_hidden = []
        all_encoder_outputs = []
        for h in beams:
            all_sec_hidden.append(h.sec_hidden)
            all_encoder_outputs.append(h.encoder_outputs)
            
        sec_hidden_stack = torch.cat(all_sec_hidden,dim=1) # (enclayer*direc,Beam,H)
        encoder_outputs_stack = torch.cat(all_encoder_outputs,dim=1) # (secL,Beam,H)
        
        extend_inp = extend_inp.repeat(1,1,config.beam_size).squeeze(0)
        results = []
        steps = 0
        while steps < config.MAX_LENGTH and len(results) < config.beam_size:
            latest_tokens = [[h.latest_token if h.latest_token < self.model.vocab_size else config.UNK_token for h in beams]]
            latest_tokens = torch.tensor(latest_tokens,device=config.device, dtype=torch.long) # (1,Beam)
            
            all_decoder_hidden = []
            
            for h in beams:
                all_decoder_hidden.append(h.decoder_hidden)
            
            decoder_hidden_stack = torch.cat(all_decoder_hidden,dim=1)# (declayer,Beam,H)
            
            probs, dec_hiddens = self.model.decoder(latest_tokens,sec_hidden_stack[:,:len(beams)],
            decoder_hidden_stack,encoder_outputs_stack[:,:len(beams)],max_oov_length,extend_inp[:,:len(beams)])#(Beam,Vocab) (layer,Beam,H)

            log_probs = torch.log(probs)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size)

            all_beams = []
            num_old_beams = 1 if steps == 0 else len(beams)

            for i in range(num_old_beams):
                h = beams[i]
                decoder_hidden = dec_hiddens[:,i].unsqueeze(1)
                sec_hidden = sec_hidden_stack[:,i].unsqueeze(1)
                encoder_outputs = encoder_outputs_stack[:,i].unsqueeze(1)
                

                for j in range(config.beam_size):
                    new_beam = h.extend(token=topk_ids[i,j].item(),
                        log_prob=topk_log_probs[i,j].item(),
                        sec_hidden=sec_hidden,
                        decoder_hidden=decoder_hidden,
                        encoder_outputs=encoder_outputs
                        )
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == config.EOS_token:
                    results.append(h)
                else:
                    beams.append(h)

                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break 

            steps += 1

        if len(results) == 0:
            results = beams

        best_beam= self.sort_beams(results)[0]
        return best_beam.tokens[1:],best_beam.avg_log_prob #(1,L) (1,)