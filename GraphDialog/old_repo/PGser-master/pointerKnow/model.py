import json 
import os
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch import optim
import torch.nn.functional as F
from train import maskNLLLoss
import csv
import pandas as pd
import re
import unicodedata
import itertools
import random
import config


class EncoderRNN(nn.Module):
    def __init__(self, embedding_size,hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.embedding_size = embedding_size

        self.gru = nn.GRU(embedding_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq) # (L,B,E)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)# (L,B,2*H)  (layer*2,B,H)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs) 
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] #(L,B,H)
        # Return output and final hidden state
        return outputs, hidden

# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output): #(1,B,H) (L,B,H)
        return torch.sum(hidden * encoder_output, dim=2) #(L,B)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t() #(B,L)

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1) #(B,1,L)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, embedding_size, encoder_n_layers, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.encoder_n_layers = encoder_n_layers

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size+hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.genprob = nn.Linear(hidden_size*3+embedding_size,1)

        self.attn = Attn(attn_model, hidden_size)
        self.secAttn = Attn(attn_model, hidden_size)

    def forward(self, input_step, sec_hidden, last_hidden, encoder_outputs, max_oov_length, extend_inp):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word  (layer*direc,B*3,H)
        embedded = self.embedding(input_step) #(1,B,E) 
        embedded = self.embedding_dropout(embedded)
        
        sec_hidden = sec_hidden.view(-1,embedded.shape[1], self.hidden_size) #(2*encoderlayer*3,B,H)
        sec_attn_weights = self.secAttn(last_hidden[-1].unsqueeze(0), sec_hidden) #(B,1,2*encoderlayer*3)
        sec_hidden_context = sec_attn_weights.bmm(sec_hidden.transpose(0,1))# (B,1,2*encoderlayer*3) * (B,2*encoderlayer*3,H) = (B,1,H)
        sec_hidden_context = sec_hidden_context.transpose(0,1)


        # Forward through unidirectional GRU
        rnn_input = torch.cat((embedded,sec_hidden_context),2) #融合section与word (1,B,E+H)
        rnn_output, hidden = self.gru(rnn_input, last_hidden) #(1,B,H) (layer,B,H)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs) #(B,1,L)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))# (B,1,L) * (B,L,H) = (B,1,H)
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0) #(B,H)
        context = context.squeeze(1) #(B,H)
        concat_input = torch.cat((rnn_output, context), 1) #(B,2*H)
        concat_output = torch.tanh(self.concat(concat_input)) #(B,H)
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)#(B,Out)
        output = F.softmax(output, dim=1)#(B,Out)

        if config.pointer_gen:
            genprob_input = torch.cat((embedded.squeeze(0),sec_hidden_context.squeeze(0),context,hidden.squeeze(0)),dim=1) #(B,E+3*H)
            p_gen = F.sigmoid(self.genprob(genprob_input)) #(B,1)

            if max_oov_length > 0:
                extra_zeros = torch.zeros((output.shape[0],max_oov_length),device=config.device)
                output = torch.cat([output,extra_zeros],dim=1)
            output_ = p_gen*output
            attn_weights_ = (1-p_gen)*attn_weights.squeeze(1)
#             print('extend size:',extend_inp.size())
#             print('output size:', output.size())
#             print('attention size:', attn_weights_.size())
            output_ = output_.scatter_add(1,extend_inp.transpose(0,1),attn_weights_)
            
            return output_, hidden

        # Return output and final hidden state
        return output, hidden #(B,Out) (layer,B,H)

class Copy_seq2seq(nn.Module):
    """docstring for Copy_seq2seq"""
    def __init__(self, vocab_size,embedding_size,hidden_size,encoder_num_layers=1,decoder_num_layers=1,dropout=0.0,
        attn_model='dot'):
        super(Copy_seq2seq, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.dropout = dropout
        self.attn_model = attn_model

        # Initialize word embeddings
        embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        # Initialize encoder & decoder models
        self.encoder = EncoderRNN(self.embedding_size,self.hidden_size, 
            embedding, self.encoder_num_layers, self.dropout)
        self.sec_encoder = EncoderRNN(self.embedding_size,self.hidden_size, 
            embedding, self.encoder_num_layers, self.dropout)
        self.decoder = LuongAttnDecoderRNN(self.attn_model, embedding,
            self.embedding_size, self.encoder_num_layers,self.hidden_size, self.vocab_size,
            self.decoder_num_layers, self.dropout)
    
    def encode(self, data_batch):
        section_variable, sec_lengths, input_variable, lengths,_,_,_,_,_,_,_ = data_batch
        
        

        # Set device options
        input_variable = input_variable.to(config.device)
        lengths = lengths.to(config.device)
        section_variable = section_variable.to(config.device)
        sec_lengths = sec_lengths.to(config.device)

        #将doc按长度降序排列，并保存让其恢复原样的sec_idx
        sec_lengths,idx1 = torch.sort(sec_lengths,descending=True)
        section_variable = section_variable.index_select(1,idx1)
        _,sec_idx = torch.sort(idx1)
        # Run a training iteration with batch

        
        encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths) # (L,B,H)  (layer*direc,B,H)
        
        # Forward pass through encoder
        sec_outputs, sec_hidden = self.sec_encoder(section_variable, sec_lengths) # (secL,B*3,H)  (layer*direc,B*3,H)
        sec_hidden = sec_hidden.index_select(1,sec_idx) #调整回按utter长度排序的batch内顺序

        
        return sec_hidden, encoder_outputs, encoder_hidden

    def decode(self, data_batch,sec_hidden,encoder_outputs,encoder_hidden):
        _,_,_,_,_,max_oov_length,extend_inp,target_variable, mask, max_target_len,extend_output = data_batch

        target_variable = target_variable.to(config.device)
        mask = mask.to(config.device)
        extend_inp = extend_inp.to(config.device)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[config.SOS_token for _ in range(config.batch_size)]]) # (1,B)
        decoder_input = decoder_input.to(config.device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[-self.decoder_num_layers:] #（layer,B,H)

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < config.teacher_forcing_ratio else False
        
        # Initialize variables
        if config.pointer_gen:
            outputs = target_variable.new_zeros(
                size=(max_target_len,config.batch_size,self.vocab_size+max_oov_length),
                dtype=torch.float)
        else:
            outputs = target_variable.new_zeros(
                size=(max_target_len,config.batch_size,self.vocab_size),
                dtype=torch.float)

        # Forward batch of sequences through decoder one time step at a time
        
        for t in range(max_target_len):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, sec_hidden, decoder_hidden, encoder_outputs,max_oov_length,extend_inp
            )
            if use_teacher_forcing:
                # Teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
            else:
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] if topi[i][0] < self.vocab_size else config.UNK_token for i in range(config.batch_size)]])
                decoder_input = decoder_input.to(config.device)

            outputs[t] = decoder_output

        return outputs

    def forward(self, data_batch):
        sec_hidden,encoder_outputs,encoder_hidden = self.encode(data_batch)
#         print(sec_hidden.size())
        out_prop_dist = self.decode(data_batch,sec_hidden,encoder_outputs,encoder_hidden)
        return out_prop_dist

    def compute_loss(self, out_prop_dist, target_variable, mask):
        loss = 0
        n_totals = 0
        print_losses = []
        # Calculate and accumulate loss
        for t in range(len(out_prop_dist)):
            mask_loss, nTotal = maskNLLLoss(out_prop_dist[t], target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

        return loss, sum(print_losses)/n_totals
    
    def iterate(self, data_batch, optimizer = None, grad_clip=None,is_training=False):
        
        out_prop_dist = self.forward(data_batch)
        _,_,_,_,_,max_oov_length,extend_inp,target_variable, mask, max_target_len,extend_output = data_batch
        if config.pointer_gen:
            target_variable = extend_output.to(config.device)
        else:
            target_variable = target_variable.to(config.device)
        mask = mask.to(config.device)
        loss, avg_losses = self.compute_loss(out_prop_dist,target_variable,mask)

        if torch.isnan(loss):
            raise ValueError("nan loss encountered")

        if is_training:
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                clip_grad_norm_(parameters=self.parameters(),
                                max_norm=grad_clip)
            optimizer.step()

        return avg_losses

    def save(self, filename):
        """
        save
        """
        torch.save(self.state_dict(), filename)
        print("Saved model state to '{}'!".format(filename))

    def load(self, filename):
        """
        load
        """
        if os.path.isfile(filename):
            state_dict = torch.load(
                filename, map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict, strict=False)
            print("Loaded model state from '{}'".format(filename))
        else:
            print("Invalid model state file: '{}'".format(filename))