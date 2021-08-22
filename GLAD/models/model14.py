import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
import logging
import os
import re
import json
from collections import defaultdict
from pprint import pformat



# def position_encoding_init(n_position, d_pos_vec):
#     ''' Init the sinusoid position encoding table '''

#     # keep dim 0 for padding token position encoding zero vector
#     position_enc = np.array([
#         [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
#         if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

#     position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
#     position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
#     return torch.from_numpy(position_enc).type(torch.FloatTensor).cuda()


def pad(seqs, emb, device, pad=0):
    lens = [len(s) for s in seqs]
    # if not lens:
    #     print(seqs)
    max_len = max(lens)
    padded = torch.LongTensor([s + (max_len-l) * [pad] for s, l in zip(seqs, lens)])
    return emb(padded.to(device)), lens


def run_rnn(rnn, inputs, lens):
    # sort by lens
    order = np.argsort(lens)[::-1].tolist()
    reindexed = inputs.index_select(0, inputs.data.new(order).long())
    reindexed_lens = [lens[i] for i in order]
    packed = nn.utils.rnn.pack_padded_sequence(reindexed, reindexed_lens, batch_first=True)
    outputs, _ = rnn(packed)
    padded, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, padding_value=0.)
    reverse_order = np.argsort(order).tolist()
    recovered = padded.index_select(0, inputs.data.new(reverse_order).long())
    return recovered


def attend(seq, cond, lens):
    """
    attend over the sequences `seq` using the condition `cond`.
    """
    if cond.ndimension() < 3:
        scores = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
    else:
        scores = cond.expand_as(seq).mul(seq).sum(2)
    max_len = max(lens)
    for i, l in enumerate(lens):
        if l < max_len:
            scores.data[i, l:] = -np.inf
    scores = F.softmax(scores, dim=1)
    context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
    return context, scores


class FixedEmbedding(nn.Embedding):
    """
    this is the same as `nn.Embedding` but detaches the result from the graph and has dropout after lookup.
    """

    def __init__(self, *args, dropout=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = dropout

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        out.detach_()
        return F.dropout(out, self.dropout, self.training)


class SelfAttention_gce(nn.Module):

    def __init__(self, dhid, dropout=0.):
        super().__init__()
        self.conv = nn.Conv1d(2 * dhid, 1, 5, padding=2)
        #self.dropout = nn.Dropout(dropout)
        #self.layer_norm = nn.LayerNorm(self.dv)

    def forward(self, inp, lens, cond):
        batch, seq_len, d_feat = inp.size()
        concat = torch.cat((cond.unsqueeze(0).expand_as(inp), inp), dim=2)
        attention = self.conv(concat.transpose(2, 1))
        scores = F.softmax(attention, dim=2)
        context = scores.bmm(inp) 
        return context

def run_cnn_(self,inputs,inputs_,x_new = False,Gate = True,SameKernel = False,Layers = 2,isResNet = True):
    inputs = inputs.permute(0, 2, 1)



    #out = self.conv1_slot(inputs)
    out = [conv(inputs) for conv in self.convs]
    out = torch.cat(out, dim=1)
    '''
    inputs          torch.Size([50, 800, 30])
    inputs_         torch.Size([50, 30, 400])
    out             torch.Size([50, 400, 30])
    '''

    #不经过残差层
    if isResNet == False:
        outputs = out.transpose(1,2)+inputs_
        return outputs

    #无门控一层残差层    
    elif isResNet == True and Gate == False and Layers == 1:
        out_ = out.transpose(1,2)+inputs_
        out = self.conv1(out_.permute(0, 2, 1))
        outputs = out.transpose(1,2) + out_
        return outputs

    #无门控两层残差层
    elif isResNet == True and Gate == False and Layers == 2:

        intputs_resNet1 = out.transpose(1,2)+inputs_
        outputs_resNet1 = self.conv1(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1 = outputs_resNet1.transpose(1,2) + intputs_resNet1

        intputs_resNet2 = outputs_resNet1
        outputs_resNet2 = self.conv1(intputs_resNet2.permute(0, 2, 1))
        outputs_resNet2 = outputs_resNet2.transpose(1,2) + intputs_resNet2


        outputs = outputs_resNet2
        return outputs

    #同核1层    
    elif isResNet == True and Gate == True and Layers == 1 and SameKernel == True:

        intputs_resNet1 = out.transpose(1,2)+inputs_
        outputs_resNet1_A = self.conv1(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_B = self.conv1(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_A2 = outputs_resNet1_A.contiguous().view(-1, outputs_resNet1_A.size(2))
        outputs_resNet1_B2 = outputs_resNet1_B.contiguous().view(-1, outputs_resNet1_B.size(2))
        attn = torch.mul(outputs_resNet1_A2, self.softmax(outputs_resNet1_B2))
        outputs_resNet1 = attn.view(outputs_resNet1_A.size(0), outputs_resNet1_A.size(1), -1)
        outputs_resNet1 = outputs_resNet1.transpose(1,2) + intputs_resNet1



        outputs = outputs_resNet1
        return outputs
    
    #同核两层
    elif isResNet == True and Gate == True and Layers == 2 and SameKernel == True:

        intputs_resNet1 = out.transpose(1,2)+inputs_
        outputs_resNet1_A = self.conv1(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_B = self.conv1(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_A2 = outputs_resNet1_A.contiguous().view(-1, outputs_resNet1_A.size(2))
        outputs_resNet1_B2 = outputs_resNet1_B.contiguous().view(-1, outputs_resNet1_B.size(2))
        attn = torch.mul(outputs_resNet1_A2, self.softmax(outputs_resNet1_B2))
        outputs_resNet1 = attn.view(outputs_resNet1_A.size(0), outputs_resNet1_A.size(1), -1)
        outputs_resNet1 = outputs_resNet1.transpose(1,2) + intputs_resNet1

        intputs_resNet2 = outputs_resNet1
        outputs_resNet2_A = self.conv1(intputs_resNet2.permute(0, 2, 1))
        outputs_resNet2_B = self.conv1(intputs_resNet2.permute(0, 2, 1))
        outputs_resNet2_A2 = outputs_resNet2_A.contiguous().view(-1, outputs_resNet2_A.size(2))
        outputs_resNet2_B2 = outputs_resNet2_B.contiguous().view(-1, outputs_resNet2_B.size(2))
        attn = torch.mul(outputs_resNet2_A2, self.softmax(outputs_resNet2_B2))
        outputs_resNet2 = attn.view(outputs_resNet2_A.size(0), outputs_resNet2_A.size(1), -1)
        outputs_resNet2 = outputs_resNet2.transpose(1,2) + intputs_resNet2

        outputs = outputs_resNet2
        return outputs
    
    #异核1层
    elif isResNet == True and Gate == True and Layers == 1 and SameKernel == False:

        intputs_resNet1 = out.transpose(1,2)+inputs_
        outputs_resNet1_A = self.conv1(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_B = self.conv3(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_A2 = outputs_resNet1_A.contiguous().view(-1, outputs_resNet1_A.size(2))
        outputs_resNet1_B2 = outputs_resNet1_B.contiguous().view(-1, outputs_resNet1_B.size(2))
        attn = torch.mul(outputs_resNet1_A2, self.softmax(outputs_resNet1_B2))
        outputs_resNet1 = attn.view(outputs_resNet1_A.size(0), outputs_resNet1_A.size(1), -1)
        outputs_resNet1 = outputs_resNet1.transpose(1,2) + intputs_resNet1

        outputs = outputs_resNet1
        return outputs

    #异核2层
    elif isResNet == True and Gate == True and Layers == 2 and SameKernel == False:

        intputs_resNet1 = out.transpose(1,2)+inputs_
        outputs_resNet1_A = self.conv1(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_B = self.conv3(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_A2 = outputs_resNet1_A.contiguous().view(-1, outputs_resNet1_A.size(2))
        outputs_resNet1_B2 = outputs_resNet1_B.contiguous().view(-1, outputs_resNet1_B.size(2))
        attn = torch.mul(outputs_resNet1_A2, self.softmax(outputs_resNet1_B2))
        outputs_resNet1 = attn.view(outputs_resNet1_A.size(0), outputs_resNet1_A.size(1), -1)
        outputs_resNet1 = outputs_resNet1.transpose(1,2) + intputs_resNet1

        intputs_resNet2 = outputs_resNet1
        outputs_resNet2_A = self.conv1(intputs_resNet2.permute(0, 2, 1))
        outputs_resNet2_B = self.conv3(intputs_resNet2.permute(0, 2, 1))
        outputs_resNet2_A2 = outputs_resNet2_A.contiguous().view(-1, outputs_resNet2_A.size(2))
        outputs_resNet2_B2 = outputs_resNet2_B.contiguous().view(-1, outputs_resNet2_B.size(2))
        attn = torch.mul(outputs_resNet2_A2, self.softmax(outputs_resNet2_B2))
        outputs_resNet2 = attn.view(outputs_resNet2_A.size(0), outputs_resNet2_A.size(1), -1)
        outputs_resNet2 = outputs_resNet2.transpose(1,2) + intputs_resNet2

        outputs = outputs_resNet2
        return outputs

    else:
        exit()
        for i in range(Layers):
            out_A = self.conv1(out)
            out_B = self.conv1(out)
            A2 = out_A.contiguous().view(-1, out_A.size(2))
            B2 = out_B.contiguous().view(-1, out_B.size(2))
            attn = torch.mul(A2, self.softmax(B2)) 
            out_ = attn.view(out_A.size(0), out_A.size(1), -1)       #torch.Size([50, 400, 30])
            out = out+out_

        outputs = out.transpose(1,2)+inputs_
        return outputs

def run_cnn(self,inputs,inputs_,Layers = 1,isdenesNet = False):
    inputs = inputs.permute(0, 2, 1)

    #out = self.conv1_slot(inputs)
    out = [conv(inputs) for conv in self.convs]
    out = torch.cat(out, dim=1)
    '''
    inputs          torch.Size([50, 800, 30])
    inputs_         torch.Size([50, 30, 400])
    out             torch.Size([50, 400, 30])
    '''
    #异核1层
    '''
    if Layers ==1 and isdenesNet == False:

        intputs_resNet1 = out.transpose(1,2)+inputs_
        outputs_resNet1_A = self.conv1(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_B = self.conv3(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_A2 = outputs_resNet1_A.contiguous().view(-1, outputs_resNet1_A.size(2))
        outputs_resNet1_B2 = outputs_resNet1_B.contiguous().view(-1, outputs_resNet1_B.size(2))
        attn = torch.mul(outputs_resNet1_A2, self.softmax(outputs_resNet1_B2))
        outputs_resNet1 = attn.view(outputs_resNet1_A.size(0), outputs_resNet1_A.size(1), -1)
        outputs_resNet1 = outputs_resNet1.transpose(1,2) + intputs_resNet1


        outputs = outputs_resNet1
        return outputs
    '''
    if Layers == 1 and  isdenesNet == False:
        intputs_resNet1 = out.transpose(1,2)+inputs_
        outputs_resNet1_A = self.conv1(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_B = self.conv3(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_C = self.conv5(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_A2 = outputs_resNet1_A.contiguous().view(-1, outputs_resNet1_A.size(2))
        outputs_resNet1_B2 = outputs_resNet1_B.contiguous().view(-1, outputs_resNet1_B.size(2))
        outputs_resNet1_C2 = outputs_resNet1_C.contiguous().view(-1, outputs_resNet1_C.size(2))
        exit()


        attn13 = torch.mul(outputs_resNet1_A2, self.softmax(outputs_resNet1_B2))
        # attn15 = torch.mul(outputs_resNet1_B2, self.softmax(outputs_resNet1_C2))
        # attn35 = torch.mul(outputs_resNet1_C2, self.softmax(outputs_resNet1_A2))

        # attn = torch.cat((attn13,attn15,attn35),dim =2)
        # attn  = self.attention(attn)


        outputs_resNet1 = attn.view(outputs_resNet1_A.size(0), outputs_resNet1_A.size(1), -1)
        outputs_resNet1 = outputs_resNet1.transpose(1,2) + intputs_resNet1


        outputs = outputs_resNet1
        return outputs



    #异核2层
    elif Layers == 2 and isdenesNet == False:

        intputs_resNet1 = out.transpose(1,2)+inputs_
        outputs_resNet1_A = self.conv1(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_B = self.conv3(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_A2 = outputs_resNet1_A.contiguous().view(-1, outputs_resNet1_A.size(2))
        outputs_resNet1_B2 = outputs_resNet1_B.contiguous().view(-1, outputs_resNet1_B.size(2))
        attn = torch.mul(outputs_resNet1_A2, self.softmax(outputs_resNet1_B2))
        outputs_resNet1 = attn.view(outputs_resNet1_A.size(0), outputs_resNet1_A.size(1), -1)
        outputs_resNet1 = outputs_resNet1.transpose(1,2) + intputs_resNet1

        intputs_resNet2 = outputs_resNet1
        outputs_resNet2_A = self.conv1(intputs_resNet2.permute(0, 2, 1))
        outputs_resNet2_B = self.conv3(intputs_resNet2.permute(0, 2, 1))
        outputs_resNet2_A2 = outputs_resNet2_A.contiguous().view(-1, outputs_resNet2_A.size(2))
        outputs_resNet2_B2 = outputs_resNet2_B.contiguous().view(-1, outputs_resNet2_B.size(2))
        attn = torch.mul(outputs_resNet2_A2, self.softmax(outputs_resNet2_B2))
        outputs_resNet2 = attn.view(outputs_resNet2_A.size(0), outputs_resNet2_A.size(1), -1)
        outputs_resNet2 = outputs_resNet2.transpose(1,2) + intputs_resNet2

        outputs = outputs_resNet2
        return outputs
    #relu
    elif Layers == 1 and isdenesNet == True:
        intputs_resNet1 = out.transpose(1,2)+inputs_
        outputs_resNet1_A = self.conv1(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_B = self.conv3(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_A = self.relu(outputs_resNet1_A)
        outputs_resNet1_B = self.relu(outputs_resNet1_B)
        outputs_resNet1_A2 = outputs_resNet1_A.contiguous().view(-1, outputs_resNet1_A.size(2))
        outputs_resNet1_B2 = outputs_resNet1_B.contiguous().view(-1, outputs_resNet1_B.size(2))
        attn = torch.mul(outputs_resNet1_A2, self.softmax(outputs_resNet1_B2))
        outputs_resNet1 = attn.view(outputs_resNet1_A.size(0), outputs_resNet1_A.size(1), -1)
        outputs_resNet1 = outputs_resNet1.transpose(1,2) + intputs_resNet1
        outputs_resNet1 = self.relu(outputs_resNet1)

        outputs = outputs_resNet1
        return outputs

    #relu
    elif Layers ==2 and isdenesNet == True:
        intputs_resNet1 = out.transpose(1,2)+inputs_
        outputs_resNet1_A = self.conv1(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_B = self.conv3(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_A = self.relu(outputs_resNet1_A)
        outputs_resNet1_B = self.relu(outputs_resNet1_B)
        outputs_resNet1_A2 = outputs_resNet1_A.contiguous().view(-1, outputs_resNet1_A.size(2))
        outputs_resNet1_B2 = outputs_resNet1_B.contiguous().view(-1, outputs_resNet1_B.size(2))
        attn = torch.mul(outputs_resNet1_A2, self.softmax(outputs_resNet1_B2))
        outputs_resNet1 = attn.view(outputs_resNet1_A.size(0), outputs_resNet1_A.size(1), -1)
        outputs_resNet1 = outputs_resNet1.transpose(1,2) + intputs_resNet1
        outputs_resNet1 = self.relu(outputs_resNet1)

        intputs_resNet2 = outputs_resNet1
        outputs_resNet2_A = self.conv1(intputs_resNet2.permute(0, 2, 1))
        outputs_resNet2_B = self.conv3(intputs_resNet2.permute(0, 2, 1))
        outputs_resNet2_A = self.relu(outputs_resNet2_A)
        outputs_resNet2_B = self.relu(outputs_resNet2_B)
        outputs_resNet2_A2 = outputs_resNet2_A.contiguous().view(-1, outputs_resNet2_A.size(2))
        outputs_resNet2_B2 = outputs_resNet2_B.contiguous().view(-1, outputs_resNet2_B.size(2))
        attn = torch.mul(outputs_resNet2_A2, self.softmax(outputs_resNet2_B2))
        outputs_resNet2 = attn.view(outputs_resNet2_A.size(0), outputs_resNet2_A.size(1), -1)
        outputs_resNet2 = outputs_resNet2.transpose(1,2) + intputs_resNet2
        outputs_resNet2 = self.relu(outputs_resNet2)

        outputs = outputs_resNet2
        return outputs
    
    elif Layers ==3 and isdenesNet == False:
        intputs_resNet1 = out.transpose(1,2)+inputs_


        outputs_resNet1_A = self.conv1(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_B = self.conv3(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_A = self.relu(outputs_resNet1_A)
        outputs_resNet1_B = self.relu(outputs_resNet1_B)
        outputs_resNet1_A2 = outputs_resNet1_A.contiguous().view(-1, outputs_resNet1_A.size(2))
        outputs_resNet1_B2 = outputs_resNet1_B.contiguous().view(-1, outputs_resNet1_B.size(2))
        attn = torch.mul(outputs_resNet1_A2, self.softmax(outputs_resNet1_B2))
        outputs_resNet1 = attn.view(outputs_resNet1_A.size(0), outputs_resNet1_A.size(1), -1)
        outputs_resNet1 = outputs_resNet1.transpose(1,2) + intputs_resNet1

        intputs_resNet2 = outputs_resNet1
        outputs_resNet2_A = self.conv1(intputs_resNet2.permute(0, 2, 1))
        outputs_resNet2_B = self.conv3(intputs_resNet2.permute(0, 2, 1))
        outputs_resNet2_A = self.relu(outputs_resNet2_A)
        outputs_resNet2_B = self.relu(outputs_resNet2_B)
        outputs_resNet2_A2 = outputs_resNet2_A.contiguous().view(-1, outputs_resNet2_A.size(2))
        outputs_resNet2_B2 = outputs_resNet2_B.contiguous().view(-1, outputs_resNet2_B.size(2))
        attn = torch.mul(outputs_resNet2_A2, self.softmax(outputs_resNet2_B2))
        outputs_resNet2 = attn.view(outputs_resNet2_A.size(0), outputs_resNet2_A.size(1), -1)
        outputs_resNet2 = outputs_resNet2.transpose(1,2) + intputs_resNet2

        inputs_resNet3 = outputs_resNet2
        outputs_resNet3_A = self.conv1(inputs_resNet3.permute(0, 2, 1))
        outputs_resNet3_B = self.conv3(inputs_resNet3.permute(0, 2, 1))
        outputs_resNet3_A = self.relu(outputs_resNet3_A)
        outputs_resNet3_B = self.relu(outputs_resNet3_B)
        outputs_resNet3_A2 = outputs_resNet3_A.contiguous().view(-1, outputs_resNet3_A.size(2))
        outputs_resNet3_B2 = outputs_resNet3_B.contiguous().view(-1, outputs_resNet3_B.size(2))
        attn = torch.mul(outputs_resNet3_A2, self.softmax(outputs_resNet3_B2))
        outputs_resNet3 = attn.view(outputs_resNet3_A.size(0), outputs_resNet3_A.size(1), -1)
        outputs_resNet3 = outputs_resNet3.transpose(1,2) + inputs_resNet3



        outputs = outputs_resNet3
        return outputs

    elif Layers ==3 and isdenesNet == True:
        intputs_resNet1 = out.transpose(1,2)+inputs_


        outputs_resNet1_A = self.conv1(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_B = self.conv3(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_A = self.relu(outputs_resNet1_A)
        outputs_resNet1_B = self.relu(outputs_resNet1_B)
        outputs_resNet1_A2 = outputs_resNet1_A.contiguous().view(-1, outputs_resNet1_A.size(2))
        outputs_resNet1_B2 = outputs_resNet1_B.contiguous().view(-1, outputs_resNet1_B.size(2))
        attn = torch.mul(outputs_resNet1_A2, self.softmax(outputs_resNet1_B2))
        outputs_resNet1 = attn.view(outputs_resNet1_A.size(0), outputs_resNet1_A.size(1), -1)
        outputs_resNet1 = outputs_resNet1.transpose(1,2) + intputs_resNet1

        intputs_resNet2 = outputs_resNet1
        outputs_resNet2_A = self.conv1(intputs_resNet2.permute(0, 2, 1))
        outputs_resNet2_B = self.conv3(intputs_resNet2.permute(0, 2, 1))
        outputs_resNet2_A = self.relu(outputs_resNet2_A)
        outputs_resNet2_B = self.relu(outputs_resNet2_B)
        outputs_resNet2_A2 = outputs_resNet2_A.contiguous().view(-1, outputs_resNet2_A.size(2))
        outputs_resNet2_B2 = outputs_resNet2_B.contiguous().view(-1, outputs_resNet2_B.size(2))
        attn = torch.mul(outputs_resNet2_A2, self.softmax(outputs_resNet2_B2))
        outputs_resNet2 = attn.view(outputs_resNet2_A.size(0), outputs_resNet2_A.size(1), -1)
        outputs_resNet2 = outputs_resNet2.transpose(1,2) + intputs_resNet2

        inputs_resNet3 = outputs_resNet2 + outputs_resNet1
        outputs_resNet3_A = self.conv1(inputs_resNet3.permute(0, 2, 1))
        outputs_resNet3_B = self.conv3(inputs_resNet3.permute(0, 2, 1))
        outputs_resNet3_A = self.relu(outputs_resNet3_A)
        outputs_resNet3_B = self.relu(outputs_resNet3_B)
        outputs_resNet3_A2 = outputs_resNet3_A.contiguous().view(-1, outputs_resNet3_A.size(2))
        outputs_resNet3_B2 = outputs_resNet3_B.contiguous().view(-1, outputs_resNet3_B.size(2))
        attn = torch.mul(outputs_resNet3_A2, self.softmax(outputs_resNet3_B2))
        outputs_resNet3 = attn.view(outputs_resNet3_A.size(0), outputs_resNet3_A.size(1), -1)
        outputs_resNet3 = outputs_resNet3.transpose(1,2) + inputs_resNet3



        outputs = outputs_resNet3
        return outputs
    
    elif Layers == 4 and isdenesNet == False:
        intputs_resNet1 = out.transpose(1,2)+inputs_


        outputs_resNet1_A = self.conv1(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_B = self.conv3(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_A = self.relu(outputs_resNet1_A)
        outputs_resNet1_B = self.relu(outputs_resNet1_B)
        outputs_resNet1_A2 = outputs_resNet1_A.contiguous().view(-1, outputs_resNet1_A.size(2))
        outputs_resNet1_B2 = outputs_resNet1_B.contiguous().view(-1, outputs_resNet1_B.size(2))
        attn = torch.mul(outputs_resNet1_A2, self.softmax(outputs_resNet1_B2))
        outputs_resNet1 = attn.view(outputs_resNet1_A.size(0), outputs_resNet1_A.size(1), -1)
        outputs_resNet1 = outputs_resNet1.transpose(1,2) + intputs_resNet1

        intputs_resNet2 = outputs_resNet1
        outputs_resNet2_A = self.conv1(intputs_resNet2.permute(0, 2, 1))
        outputs_resNet2_B = self.conv3(intputs_resNet2.permute(0, 2, 1))
        outputs_resNet2_A = self.relu(outputs_resNet2_A)
        outputs_resNet2_B = self.relu(outputs_resNet2_B)
        outputs_resNet2_A2 = outputs_resNet2_A.contiguous().view(-1, outputs_resNet2_A.size(2))
        outputs_resNet2_B2 = outputs_resNet2_B.contiguous().view(-1, outputs_resNet2_B.size(2))
        attn = torch.mul(outputs_resNet2_A2, self.softmax(outputs_resNet2_B2))
        outputs_resNet2 = attn.view(outputs_resNet2_A.size(0), outputs_resNet2_A.size(1), -1)
        outputs_resNet2 = outputs_resNet2.transpose(1,2) + intputs_resNet2

        inputs_resNet3 = outputs_resNet2 
        outputs_resNet3_A = self.conv1(inputs_resNet3.permute(0, 2, 1))
        outputs_resNet3_B = self.conv3(inputs_resNet3.permute(0, 2, 1))
        outputs_resNet3_A = self.relu(outputs_resNet3_A)
        outputs_resNet3_B = self.relu(outputs_resNet3_B)
        outputs_resNet3_A2 = outputs_resNet3_A.contiguous().view(-1, outputs_resNet3_A.size(2))
        outputs_resNet3_B2 = outputs_resNet3_B.contiguous().view(-1, outputs_resNet3_B.size(2))
        attn = torch.mul(outputs_resNet3_A2, self.softmax(outputs_resNet3_B2))
        outputs_resNet3 = attn.view(outputs_resNet3_A.size(0), outputs_resNet3_A.size(1), -1)
        outputs_resNet3 = outputs_resNet3.transpose(1,2) + inputs_resNet3

        inputs_resNet4 = outputs_resNet3
        outputs_resNet4_A = self.conv1(inputs_resNet4.permute(0, 2, 1))
        outputs_resNet4_B = self.conv3(inputs_resNet4.permute(0, 2, 1))
        outputs_resNet4_A = self.relu(outputs_resNet4_A)
        outputs_resNet4_B = self.relu(outputs_resNet4_B)
        outputs_resNet4_A2 = outputs_resNet4_A.contiguous().view(-1, outputs_resNet4_A.size(2))
        outputs_resNet4_B2 = outputs_resNet4_B.contiguous().view(-1, outputs_resNet4_B.size(2))
        attn = torch.mul(outputs_resNet4_A2, self.softmax(outputs_resNet4_B2))
        outputs_resNet4 = attn.view(outputs_resNet4_A.size(0), outputs_resNet4_A.size(1), -1)
        outputs_resNet4 = outputs_resNet4.transpose(1,2) + inputs_resNet4



        outputs = outputs_resNet4
        return outputs
     
    elif Layers == 4 and isdenesNet == True:
        intputs_resNet1 = out.transpose(1,2)+inputs_


        outputs_resNet1_A = self.conv1(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_B = self.conv3(intputs_resNet1.permute(0, 2, 1))
        outputs_resNet1_A = self.relu(outputs_resNet1_A)
        outputs_resNet1_B = self.relu(outputs_resNet1_B)
        outputs_resNet1_A2 = outputs_resNet1_A.contiguous().view(-1, outputs_resNet1_A.size(2))
        outputs_resNet1_B2 = outputs_resNet1_B.contiguous().view(-1, outputs_resNet1_B.size(2))
        attn = torch.mul(outputs_resNet1_A2, self.softmax(outputs_resNet1_B2))
        outputs_resNet1 = attn.view(outputs_resNet1_A.size(0), outputs_resNet1_A.size(1), -1)
        outputs_resNet1 = outputs_resNet1.transpose(1,2) + intputs_resNet1

        intputs_resNet2 = outputs_resNet1
        outputs_resNet2_A = self.conv1(intputs_resNet2.permute(0, 2, 1))
        outputs_resNet2_B = self.conv3(intputs_resNet2.permute(0, 2, 1))
        outputs_resNet2_A = self.relu(outputs_resNet2_A)
        outputs_resNet2_B = self.relu(outputs_resNet2_B)
        outputs_resNet2_A2 = outputs_resNet2_A.contiguous().view(-1, outputs_resNet2_A.size(2))
        outputs_resNet2_B2 = outputs_resNet2_B.contiguous().view(-1, outputs_resNet2_B.size(2))
        attn = torch.mul(outputs_resNet2_A2, self.softmax(outputs_resNet2_B2))
        outputs_resNet2 = attn.view(outputs_resNet2_A.size(0), outputs_resNet2_A.size(1), -1)
        outputs_resNet2 = outputs_resNet2.transpose(1,2) + intputs_resNet2

        inputs_resNet3 = outputs_resNet2 + outputs_resNet1
        outputs_resNet3_A = self.conv1(inputs_resNet3.permute(0, 2, 1))
        outputs_resNet3_B = self.conv3(inputs_resNet3.permute(0, 2, 1))
        outputs_resNet3_A = self.relu(outputs_resNet3_A)
        outputs_resNet3_B = self.relu(outputs_resNet3_B)
        outputs_resNet3_A2 = outputs_resNet3_A.contiguous().view(-1, outputs_resNet3_A.size(2))
        outputs_resNet3_B2 = outputs_resNet3_B.contiguous().view(-1, outputs_resNet3_B.size(2))
        attn = torch.mul(outputs_resNet3_A2, self.softmax(outputs_resNet3_B2))
        outputs_resNet3 = attn.view(outputs_resNet3_A.size(0), outputs_resNet3_A.size(1), -1)
        outputs_resNet3 = outputs_resNet3.transpose(1,2) + inputs_resNet3

        inputs_resNet4 = outputs_resNet3 + outputs_resNet2 + outputs_resNet1
        outputs_resNet4_A = self.conv1(inputs_resNet4.permute(0, 2, 1))
        outputs_resNet4_B = self.conv3(inputs_resNet4.permute(0, 2, 1))
        outputs_resNet4_A = self.relu(outputs_resNet4_A)
        outputs_resNet4_B = self.relu(outputs_resNet4_B)
        outputs_resNet4_A2 = outputs_resNet4_A.contiguous().view(-1, outputs_resNet4_A.size(2))
        outputs_resNet4_B2 = outputs_resNet4_B.contiguous().view(-1, outputs_resNet4_B.size(2))
        attn = torch.mul(outputs_resNet4_A2, self.softmax(outputs_resNet4_B2))
        outputs_resNet4 = attn.view(outputs_resNet4_A.size(0), outputs_resNet4_A.size(1), -1)
        outputs_resNet4 = outputs_resNet4.transpose(1,2) + inputs_resNet4



        outputs = outputs_resNet4
        return outputs


    else:
        exit()
        return outputs



class GCEencoder(nn.Module):
    """
    the GCE encoder described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, din, dhid, slots, dropout=None):
        super().__init__()
        self.dropout = dropout or {}
        self.global_rnn = nn.LSTM(2 * din, dhid, bidirectional=True, batch_first=True)
        self.global_selfattn = SelfAttention_gce(2 * dhid, dropout=self.dropout.get('selfattn', 0.))
        self.slots = slots
        self.beta_raw = nn.Parameter(torch.Tensor(len(slots)))
        nn.init.uniform_(self.beta_raw, -0.01, 0.01)

        self.in_channels = 1
        # self.out_channels = self.hidden_size * 2
        self.out_channels = 400
        self.padding = (1//2, 0)
        self.stride = 1
        self.kernel_size = (1,400)
        self.kernel_size2 = (3,400)
        self.embedding_size = 400
        #self.kernel_size = 3  #tcn

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.mapping = nn.Linear(self.embedding_size, 2*self.embedding_size)
        self.attention = nn.Linear(1200,400)
        self.convs = nn.ModuleList([
            # nn.Sequential(nn.Conv1d(in_channels=800, out_channels=100, kernel_size=h),nn.ReLU(),nn.MaxPool1d(kernel_size=30-h+1))for h in [3,4,5,6]])
            nn.Sequential(nn.Conv1d(in_channels=800, out_channels=400, kernel_size=h,padding=h//2),nn.ReLU())for h in [1]])
        # self.conv1_slot = nn.Sequential(nn.Conv1d(in_channels=800, out_channels=400, kernel_size=1,padding=1//2),nn.ReLU(),)

        self.conv1_slot = nn.Sequential(nn.Conv1d(in_channels=800, out_channels=400, kernel_size=1,padding=1//2),nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=400, out_channels=400, kernel_size=1,padding=1//2),nn.ReLU(),)
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=400, out_channels=400, kernel_size=3,padding=3//2),nn.ReLU(),)
        self.conv5 = nn.Sequential(nn.Conv1d(in_channels=400, out_channels=400, kernel_size=5,padding=5//2),nn.ReLU(),)


    def beta(self, slot):
        return F.sigmoid(self.beta_raw[self.slots.index(slot)])

    def forward(self, x, x_len, slot, slot_emb, default_dropout=0.2):
        beta = self.beta(slot)
        x_new = torch.cat((slot_emb.unsqueeze(0).expand_as(x), x), dim=2)
        # global_h = run_rnn(self.global_rnn, x_new, x_len)
        # global_h = run_cnn(self,inputs=x_new,inputs_=x,isResNet = False)                        #无残差层
        # global_h = run_cnn(self,inputs=x_new,inputs_=x,isResNet=True,Gate=True,Layers=1,SameKernel=False)        #异核1层

        # global_h = run_cnn(self,inputs=x_new,inputs_=x,isResNet=True,Gate=True,Layers=2,SameKernel=False)         #异核2层
        global_h = run_cnn(self,inputs=x_new,inputs_=x,isdenesNet=True)         #异核2层

        h = F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        c = F.dropout(self.global_selfattn(h, x_len, slot_emb), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        return h, c

class GCEdecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_ =nn.Conv1d(2*400, 400, 5, padding=2)
        self.utt_scorer = nn.Linear(400, 1)
        self.attend = Attention()

    def forward(self,batch,H_utt,c_utt,H_acts,C_acts,H_vals,C_vals,utterance_len):
        y_utts = []
        q_utts = []
        for c_val in C_vals:
            c_val = c_val.squeeze(0)                                # c_val torch.Size([400])
            q_utt= self.attend(H_utt, c_val.unsqueeze(0).expand(len(batch), *c_val.size()),lens = utterance_len)
            q_utts.append(q_utt)     
        y_utts = self.utt_scorer(torch.stack(q_utts, dim=1)).squeeze(2)
        q_acts = []
        for i, C_act in enumerate( C_acts): 
            q_act = self.attend(C_act.unsqueeze(0), c_utt[i].unsqueeze(0),lens = [C_act.size(0)])            
            q_acts.append(q_act)
        y_acts = torch.cat(q_acts, dim=0).squeeze().mm(C_vals.squeeze().transpose(0, 1))
        return y_utts,y_acts

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2*400, 1, 5, padding=2)
        self.conv_ =nn.Conv1d(2*400, 400, 5, padding=2)
        self.conv_dec = nn.Conv1d(400,400,5,padding = 2)
        self.weight_multiplicative = nn.Linear(400, 400)
        self.v = nn.Parameter(torch.FloatTensor(1, 400))
        self.w1 = nn.Linear(400, 400)
        self.w2 = nn.Linear(400, 400)

    def forward(self,inp,cond,lens,attention_type = 'default'):

        # print(inp.size())
        '''
        cond.unsqueeze(1).expand_as(inp)        torch.Size([50, 30, 400])   torch.Size([1, 3, 1, 400])
        inp                                     torch.Size([50, 30, 400])   torch.Size([1, 3, 1, 400])
        '''
        if attention_type == 'default':
            if cond.ndimension() < 3:
                scores = cond.unsqueeze(1).expand_as(inp).mul(inp).sum(2)
            else:
                scores = cond.expand_as(inp).mul(inp).sum(2)
            max_len = max(lens)
            for i, l in enumerate(lens):
                if l < max_len:
                    scores.data[i, l:] = -np.inf
            scores = F.softmax(scores, dim=1)
            context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
            return context

        elif attention_type == 'self_attention':
            if cond.ndimension() < 3:
                cond_ = cond.unsqueeze(1).expand_as(inp) 
                concat  =torch.cat((cond_, inp), dim=2).transpose(1,2)  
                scores_A = self.conv_(concat).transpose(1,2)+inp
                scores_B = self.conv_(concat).transpose(1,2)+inp
                A2 = scores_A.contiguous().view(-1, scores_A.size(2))
                B2 = scores_B.contiguous().view(-1, scores_B.size(2)) 
                attn = torch.mul(A2,B2).view(scores_A.size(0), scores_A.size(1), -1)
                context = attn.mul(inp).sum(1)
                return context
            else:
                cond_ = cond.expand_as(inp)
                concat  =torch.cat((cond_, inp), dim=3).squeeze(0).transpose(1,2)
                scores_A = self.conv_(concat).transpose(1,2)+inp.squeeze(0)
                scores_B = self.conv_(concat).transpose(1,2)+inp.squeeze(0)   
                A2 = scores_A.contiguous().view(-1, scores_A.size(2)) # A2: batch * seq_len, hidden
                B2 = scores_B.contiguous().view(-1, scores_B.size(2)) # B2: batch * seq_len, hidden
                attn = torch.mul(A2, B2).view(1, -1,scores_A.size(2))
                context = attn.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
                return context

        elif attention_type == 'multiplicative_attention':
            inp_ = self.weight_multiplicative(inp)
            if cond.ndimension() < 3:
                # scores = cond.unsqueeze(1).expand_as(inp).mul(inp).sum(2)
                cond_ = cond.unsqueeze(1).expand_as(inp)
                scores = cond_.mul(inp_).sum(2)
            else:
                scores = cond.expand_as(inp).mul(inp_).sum(2)
            '''
            max_len = max(lens)
            for i, l in enumerate(lens):
                if l < max_len:
                    scores.data[i, l:] = -np.inf
            scores = F.softmax(scores, dim=1)
            '''
            context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
            return context
        
        elif attention_type == 'additive_attention':
            # addition = torch.mm(self.w1, torch.t(hidden)) + (torch.mm(self.w2, torch.t(e)))
            if cond.ndimension() < 3:
                cond_ = cond.unsqueeze(1).expand_as(inp)
                cond_w1  = self.w1(cond_)
                inp_w2   = self.w2(inp)
                addition = torch.cat((cond_w1,inp_w2),-2)
                addition_tanh = torch.tanh(addition)

                print(addition_tanh.size())
                exit()
                # print(self.v.size())
                # exit()
        elif attention_type == 'selfattention_nocnn':
            if cond.ndimension() < 3:
                cond_ = cond.unsqueeze(1).expand_as(inp) 
                concat  =torch.cat((cond_, inp), dim=1).transpose(1,2)  
                # scores_A = self.conv_(concat).transpose(1,2)+inp
                # scores_B = self.conv_(concat).transpose(1,2)+inp
                scores_A = concat
                scores_B = concat
                A2 = scores_A.contiguous().view(-1, scores_A.size(2))
                B2 = scores_B.contiguous().view(-1, scores_B.size(2)) 
                attn = torch.mul(A2,B2).view(scores_A.size(0), scores_A.size(1), -1)
                context = attn.mul(inp).sum(1)
                return context
            else:
                cond_ = cond.expand_as(inp)
                concat  =torch.cat((cond_, inp), dim=3).squeeze(0).transpose(1,2)
                scores_A = self.conv_(concat).transpose(1,2)+inp.squeeze(0)
                scores_B = self.conv_(concat).transpose(1,2)+inp.squeeze(0)   
                A2 = scores_A.contiguous().view(-1, scores_A.size(2)) # A2: batch * seq_len, hidden
                B2 = scores_B.contiguous().view(-1, scores_B.size(2)) # B2: batch * seq_len, hidden
                attn = torch.mul(A2, B2).view(1, -1,scores_A.size(2))
                context = attn.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
                return context


class Model(nn.Module):
    """
    the GLAD scoring model described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, args, ontology, vocab):
        super().__init__()
        self.optimizer = None
        self.args = args
        self.vocab = vocab
        self.ontology = ontology
        self.emb_fixed = FixedEmbedding(len(vocab), args.demb, dropout=args.dropout.get('emb', 0.2))
        self.encoder = GCEencoder

        self.utt_encoder = self.encoder(args.demb, args.dhid, self.ontology.slots, dropout=args.dropout)
        self.act_encoder = self.encoder(args.demb, args.dhid, self.ontology.slots, dropout=args.dropout)
        self.ont_encoder = self.encoder(args.demb, args.dhid, self.ontology.slots, dropout=args.dropout)
        self.utt_scorer = nn.Linear(2 * args.dhid, 1)
        self.score_weight = nn.Parameter(torch.Tensor([0.5]))

        self.decoder = GCEdecoder()

    @property
    def device(self):
        if self.args.gpu is not None and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.args.lr)

    def load_emb(self, Eword):
        new = self.emb_fixed.weight.data.new
        self.emb_fixed.weight.data.copy_(new(Eword))

    def forward(self, batch):
        # convert to variables and look up embeddings
        eos = self.vocab.word2index('<eos>')
        ontology = {s: pad(v, self.emb_fixed, self.device, pad=eos) for s, v in self.ontology.num.items()}
        # for s,v in self.ontology.num.items():
        #     if v:
        #         ontology = {s: pad(v, self.emb_fixed, self.device, pad=eos)}

        utterance, utterance_len = pad([e.num['transcript'] for e in batch], self.emb_fixed, self.device, pad=eos)
        acts = [pad(e.num['system_acts'], self.emb_fixed, self.device, pad=eos) for e in batch]
        ys = {}
        # for each in ontology:
        #     print(each)
        # for each in self.ontology.slots:
        #     print(s)
        # exit()
        for s in self.ontology.slots:
            # for each slot, compute the scores for each value
            
            s_words = s.split()
            s_new = s_words[0]
            s_emb = self.emb_fixed(torch.cuda.LongTensor([self.vocab.word2index(s_new)]))
            H_utt, c_utt = self.utt_encoder(utterance, utterance_len, slot=s, slot_emb=s_emb)
            H_acts, C_acts = list(zip(*[self.act_encoder(a, a_len, slot=s, slot_emb=s_emb) for a, a_len in acts]))
            H_vals, C_vals = self.ont_encoder(ontology[s][0], ontology[s][1], slot=s, slot_emb=s_emb)

            y_utts,y_acts = self.decoder(batch,H_utt,c_utt,H_acts,C_acts,H_vals,C_vals,utterance_len)
            # exit()

            '''

            # compute the utterance score
            y_utts = []
            q_utts = []
            for c_val in C_vals:
                c_val = c_val.squeeze(0)
                q_utt, _ = attend(H_utt, c_val.unsqueeze(0).expand(len(batch), *c_val.size()), lens=utterance_len)
                q_utts.append(q_utt)
            y_utts = self.utt_scorer(torch.stack(q_utts, dim=1)).squeeze(2)

            # compute the previous action score
            q_acts = []
            for i, C_act in enumerate(C_acts):
                q_act, _ = attend(C_act.unsqueeze(0), c_utt[i].unsqueeze(0), lens=[C_act.size(0)])
                q_acts.append(q_act)
            
            y_acts = torch.cat(q_acts, dim=0).squeeze().mm(C_vals.squeeze().transpose(0, 1))
            '''

            # combine the scores
            ys[s] = F.sigmoid(y_utts + self.score_weight * y_acts)

        if self.training:
            # create label variable and compute loss
            labels = {s: [len(self.ontology.values[s]) * [0] for i in range(len(batch))] for s in self.ontology.slots}
            for i, e in enumerate(batch):
                for s, v in e.turn_label:
                    labels[s][i ][self.ontology.values[s].index(v)] = 1
            labels = {s: torch.Tensor(m).to(self.device) for s, m in labels.items()}
            
            loss = 0
            for s in self.ontology.slots:
                loss += F.binary_cross_entropy(ys[s], labels[s])
        else:
            loss = torch.Tensor([0]).to(self.device)
        return loss, {s: v.data.tolist() for s, v in ys.items()}

    def get_train_logger(self):
        logger = logging.getLogger('train-{}'.format(self.__class__.__name__))
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.args.dout, 'train.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def run_train(self, train, dev, args):
        track = defaultdict(list)
        iteration = 0
        best = {}
        logger = self.get_train_logger()
        if self.optimizer is None:
            self.set_optimizer()

        for epoch in range(args.epoch):
            logger.info('starting epoch {}'.format(epoch))

            # train and update parameters
            self.train()
            for batch in train.batch(batch_size=args.batch_size, shuffle=True):
                iteration += 1
                self.zero_grad()
                loss, scores = self.forward(batch)
                loss.backward()
                self.optimizer.step()
                track['loss'].append(loss.item())

            # evalute on train and dev
            summary = {'iteration': iteration, 'epoch': epoch}
            for k, v in track.items():
                summary[k] = sum(v) / len(v)
            summary.update({'eval_train_{}'.format(k): v for k, v in self.run_eval(train, args).items()})
            summary.update({'eval_dev_{}'.format(k): v for k, v in self.run_eval(dev, args).items()})

            # do early stopping saves
            stop_key = 'eval_dev_{}'.format(args.stop)
            train_key = 'eval_train_{}'.format(args.stop)
            if best.get(stop_key, 0) <= summary[stop_key]:
                best_dev = '{:f}'.format(summary[stop_key])
                best_train = '{:f}'.format(summary[train_key])
                best.update(summary)
                self.save(
                    best,
                    identifier='epoch={epoch},iter={iteration},train_{key}={train},dev_{key}={dev}'.format(
                        epoch=epoch, iteration=iteration, train=best_train, dev=best_dev, key=args.stop,
                    )
                )
                self.prune_saves()
                dev.record_preds(
                    preds=self.run_pred(dev, self.args),
                    to_file=os.path.join(self.args.dout, 'dev.pred.json'),
                )
            summary.update({'best_{}'.format(k): v for k, v in best.items()})
            logger.info(pformat(summary))
            track.clear()

    def extract_predictions(self, scores, threshold=0.5):
        batch_size = len(list(scores.values())[0])
        predictions = [set() for i in range(batch_size)]
        for s in self.ontology.slots:
            for i, p in enumerate(scores[s]):
                triggered = [(s, v, p_v) for v, p_v in zip(self.ontology.values[s], p) if p_v > threshold]
                if s == 'request':
                    # we can have multiple requests predictions
                    predictions[i] |= set([(s, v) for s, v, p_v in triggered])
                elif triggered:
                    # only extract the top inform prediction
                    sort = sorted(triggered, key=lambda tup: tup[-1], reverse=True)
                    predictions[i].add((sort[0][0], sort[0][1]))
        return predictions

    def run_pred(self, dev, args):
        self.eval()
        predictions = []
        for batch in dev.batch(batch_size=args.batch_size):
            loss, scores = self.forward(batch)
            predictions += self.extract_predictions(scores)
        return predictions

    def run_eval(self, dev, args):
        predictions = self.run_pred(dev, args)
        return dev.evaluate_preds(predictions)

    def save_config(self):
        fname = '{}/config.json'.format(self.args.dout)
        with open(fname, 'wt') as f:
            logging.info('saving config to {}'.format(fname))
            json.dump(vars(self.args), f, indent=2)

    @classmethod
    def load_config(cls, fname, ontology, **kwargs):
        with open(fname) as f:
            logging.info('loading config from {}'.format(fname))
            args = object()
            for k, v in json.load(f):
                setattr(args, k, kwargs.get(k, v))
        return cls(args, ontology)

    def save(self, summary, identifier):
        fname = '{}/{}.t7'.format(self.args.dout, identifier)
        logging.info('saving model to {}'.format(fname))
        state = {
            'args': vars(self.args),
            'model': self.state_dict(),
            'summary': summary,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, fname)

    def load(self, fname):
        logging.info('loading model from {}'.format(fname))
        state = torch.load(fname)
        self.load_state_dict(state['model'])
        self.set_optimizer()
        self.optimizer.load_state_dict(state['optimizer'])

    def get_saves(self, directory=None):
        if directory is None:
            directory = self.args.dout
        files = [f for f in os.listdir(directory) if f.endswith('.t7')]
        scores = []
        for fname in files:
            re_str = r'dev_{}=([0-9\.]+)'.format(self.args.stop)
            dev_acc = re.findall(re_str, fname)
            if dev_acc:
                score = float(dev_acc[0].strip('.'))
                scores.append((score, os.path.join(directory, fname)))
        if not scores:
            raise Exception('No files found!')
        scores.sort(key=lambda tup: tup[0], reverse=True)
        return scores

    def prune_saves(self, n_keep=5):
        scores_and_files = self.get_saves()
        if len(scores_and_files) > n_keep:
            for score, fname in scores_and_files[n_keep:]:
                os.remove(fname)

    def load_best_save(self, directory):
        if directory is None:
            directory = self.args.dout

        scores_and_files = self.get_saves(directory=directory)
        if scores_and_files:
            assert scores_and_files, 'no saves exist at {}'.format(directory)
            score, fname = scores_and_files[0]
            self.load(fname)
