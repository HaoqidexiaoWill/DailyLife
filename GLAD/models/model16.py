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
from models.TCN import TemporalConvNet


def pad(seqs, emb, device, pad=0):
    lens = [len(s) for s in seqs]
    max_len = max(lens)
    #print("len",max_len)
    padded = torch.LongTensor([s + (max_len-l) * [pad] for s, l in zip(seqs, lens)])
    # print(emb(padded.to(device)).size())
    # exit()
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

def run_tcn(self,inputs):
    outputs = inputs.transpose(1, 2)
    outputs = self.tcn(outputs)       
    outputs = outputs.transpose(1, 2)
    return outputs+inputs



def run_cnn(self,inputs,inputs_,x_new = False,Gate = True,SameKernel = False,Layers = 2,isResNet = True):
    inputs = inputs.permute(0, 2, 1)

    #out = self.conv1_slot(inputs)
    out = [conv(inputs) for conv in self.convs]
    out = torch.cat(out, dim=1)

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




def attend(seq, cond, lens):
    """
    attend over the sequences `seq` using the condition `cond`.
    """
    # print(seq.size())                   #torch.Size([50, 30, 400])
    # print(cond.size())                  #torch.Size([50, 400])
    if cond.ndimension() < 3:
        scores = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
    else:
        scores = cond.expand_as(seq).mul(seq).sum(2)                
    # print(cond.unsqueeze(1).expand_as(seq).mul(seq).size())       #torch.Size([50, 30, 400])
    # print(scores.size())                                          #torch.Size([50, 30])
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
        # print('inp_size',inp.size())                    #inp_size torch.Size([50, 30, 400])
        # print('cond_size',cond.size())                  #cond_size torch.Size([1, 400])
        # print('cat_size',torch.cat((cond.unsqueeze(0).expand_as(inp), inp), dim=2).size())
        # #cat_size torch.Size([50, 30, 800])
        concat = torch.cat((cond.unsqueeze(0).expand_as(inp), inp), dim=2)
        attention = self.conv(concat.transpose(2, 1))
        scores = F.softmax(attention, dim=2)
        context = scores.bmm(inp) 
        return context


class GCEencoder(nn.Module):
    """
    the GCE encoder described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, din, dhid, slots, dropout=None):
        super().__init__()
        self.dropout = dropout or {}
        self.global_rnn = nn.LSTM(2 * din, dhid, bidirectional=True, batch_first=True)  
        self.global_selfattn = SelfAttention_gce(2 * dhid, dropout=self.dropout.get('selfattn', 0.))
        for s in slots:
            setattr(self, '{}_cnn'.format(s),nn.Sequential(nn.Conv1d(in_channels=400, out_channels=400, kernel_size=1,padding=1//2),nn.ReLU(),))

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

        self.softmax = nn.Softmax()
        self.mapping = nn.Linear(self.embedding_size, 2*self.embedding_size)
        self.convs = nn.ModuleList([
            # nn.Sequential(nn.Conv1d(in_channels=800, out_channels=100, kernel_size=h),nn.ReLU(),nn.MaxPool1d(kernel_size=30-h+1))for h in [3,4,5,6]])
            nn.Sequential(nn.Conv1d(in_channels=800, out_channels=400, kernel_size=h,padding=h//2),nn.ReLU())for h in [1]])
        # self.conv1_slot = nn.Sequential(nn.Conv1d(in_channels=800, out_channels=400, kernel_size=1,padding=1//2),nn.ReLU(),)

        self.conv1_slot = nn.Sequential(nn.Conv1d(in_channels=800, out_channels=400, kernel_size=1,padding=1//2),nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=400, out_channels=400, kernel_size=1,padding=1//2),nn.ReLU(),)
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=400, out_channels=400, kernel_size=3,padding=3//2),nn.ReLU(),)



        self.tcn = TemporalConvNet(
            400, 
            [self.out_channels]*2, 
            3, dropout=0.2, max_length=100, attention='attention')


    def beta(self, slot):
        return F.sigmoid(self.beta_raw[self.slots.index(slot)])

    def forward(self, x, x_len, slot, slot_emb, default_dropout=0.2):
        beta = self.beta(slot)
        x_new = torch.cat((slot_emb.unsqueeze(0).expand_as(x), x), dim=2) # x utterance  x-slot type 
        #global_h = run_rnn(self.global_rnn, x_new, x_len)
        #global_h = run_tcn(self,x)

        # global_h = run_cnn(self, x_new,x)                                                                        #默认参数
        # global_h = run_cnn(self,inputs=x_new,inputs_=x,isResNet = False)                                           #无残差层
        global_h = run_cnn(self,inputs=x_new,inputs_=x,isResNet=True,Gate= False,Layers=1)                       #无门控一层残差层
        # global_h = run_cnn(self,inputs=x_new,inputs_=x,isResNet=True,Gate= False,Layers=2)                       #无门控两层残差层
        # global_h = run_cnn(self,inputs=x_new,inputs_=x,isResNet=True,Gate=True,Layers=1,SameKernel=True)         #同核1层
        # global_h = run_cnn(self,inputs=x_new,inputs_=x,isResNet=True,Gate=True,Layers=2,SameKernel=True)         #同核2层
        # global_h = run_cnn(self,inputs=x_new,inputs_=x,isResNet=True,Gate=True,Layers=1,SameKernel=False)        #异核1层
        # global_h = run_cnn(self,inputs=x_new,inputs_=x,isResNet=True,Gate=True,Layers=2,SameKernel=False)        #异核2层
        
        h = F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        c = F.dropout(self.global_selfattn(h, x_len, slot_emb), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        return h, c


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
        self.his_encoder = self.encoder(args.demb, args.dhid, self.ontology.slots, dropout=args.dropout)
        self.utt_scorer = nn.Linear(2 * args.dhid, 1)
        self.utt_scorer_his = nn.Linear(2 * args.dhid, 1)
        self.score_weight = nn.Parameter(torch.Tensor([0.5]))
        self.score_weight_his = nn.Parameter(torch.Tensor([0.5]))
        self.mapping = nn.Linear(21, 7)

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
        #print("batch",batch)
        eos = self.vocab.word2index('<eos>')
        ontology = {s: pad(v, self.emb_fixed, self.device, pad=eos) for s, v in self.ontology.num.items()}
        utterance, utterance_len = pad([e.num['transcript'] for e in batch], self.emb_fixed, self.device, pad=eos)
        acts = [pad(e.num['system_acts'], self.emb_fixed, self.device, pad=eos) for e in batch]
        history,history_len = pad([e.num['history'] for e in batch], self.emb_fixed, self.device, pad=eos)
        # print(history)
        # print(history_len)
        # print(history.size())
        # exit()
        #print(acts[0][0].size())                     #torch.Size([1, 3, 400])
        #print(utterance.size())                     #torch.Size([50, 30, 400])

        
        ys = {}
        for s in self.ontology.slots:
            # for each slot, compute the scores for each value
            
            s_words = s.split()
            s_new = s_words[0]              #attraction-area
            s_emb = self.emb_fixed(torch.cuda.LongTensor([self.vocab.word2index(s_new)]))       #torch.Size([1, 400])
            H_utt, c_utt = self.utt_encoder(utterance, utterance_len, slot=s, slot_emb=s_emb)
            H_his, c_his = self.his_encoder(history,history_len,slot = s,slot_emb = s_emb)
            H_acts, C_acts = list(zip(*[self.act_encoder(a, a_len, slot=s, slot_emb=s_emb) for a, a_len in acts]))
            H_vals, C_vals = self.ont_encoder(ontology[s][0], ontology[s][1], slot=s, slot_emb=s_emb)


            # print('H_utt size',H_utt.size())                          #H_utt size torch.Size([50, 30, 400])
            # print('c_utt size',c_utt[0].size())                       # c_utt是一个大小为50的元组，元组的size是torch.Size([1, 400])
            # print('C_acts size',len(C_acts),C_acts[0].size())         # C_acts size 50 torch.Size([1, 1, 400])
            # print('C_vals size',len(C_vals),C_vals[0].size())         # C_vals torch.Size([7, 1, 400])
            # print('H_acts',len(H_acts),H_acts[0].size())              # H_acts 50 torch.Size([1, 3, 400])
            # print('H_vals',len(H_vals),H_vals[0].size())              # H_vals 7 torch.Size([5, 400])
            # print(H_his.size())
            # print(c_his.size())
            # exit()
            # compute the utterance score
            y_utts = []
            q_utts = []
            q_hiss = []
            y_hiss = []
            for c_val in C_vals:
                c_val = c_val.squeeze(0)                                # c_val torch.Size([400])
                q_utt, _ = attend(H_utt, c_val.unsqueeze(0).expand(len(batch), *c_val.size()), lens=utterance_len)
                # c_val.unsqueeze(0).expand(len(batch), *c_val.size()   torch.Size([50, 400])
                q_utts.append(q_utt)                                    # q_utt torch.Size([50, 400])
            #print(len(q_utts))                                         # q_utts 是一个大小为7的list
            y_utts = self.utt_scorer(torch.stack(q_utts, dim=1)).squeeze(2)
            # print(torch.stack(q_utts, dim=1).size())                    # torch.Size([50, 7, 400])
            # print(self.utt_scorer(torch.stack(q_utts, dim=1)).size())   # torch.Size([50, 7, 1])

            # compute the previous action score
            q_acts = []
            for i, C_act in enumerate(C_acts):                          # 历史每一轮act 和 当前 utterance 算注意力
                q_act, _ = attend(C_act.unsqueeze(0), c_utt[i].unsqueeze(0), lens=[C_act.size(0)])
                q_acts.append(q_act)

            y_acts = torch.cat(q_acts, dim=0).squeeze().mm(C_vals.squeeze().transpose(0, 1))
            # print(torch.cat(q_acts, dim=0).squeeze().size())            # torch.Size([50, 400])
            # print(C_vals.squeeze().transpose(0, 1).size())              # torch.Size([400, 7])


            for c_val in C_vals:
                c_val = c_val.squeeze(0)
                q_his, _ = attend(H_his, c_val.unsqueeze(0).expand(len(batch), *c_val.size()), lens=history_len)
                q_hiss.append(q_his)
            y_hiss = self.utt_scorer_his(torch.stack(q_hiss, dim=1)).squeeze(2)


            y = torch.cat(([y_utts,y_acts,y_hiss]),dim =1)
            y_ =self.mapping(y)
            ys[s] = F.sigmoid(y_)
            # print(y_utts)
            # ys[s] = F.sigmoid(y_utts + self.score_weight * y_acts+self.score_weight_his*y_hiss)
            # ys[s] = F.sigmoid(y_utts + self.score_weight * y_acts)
            #ys[s] = F.sigmoid(y_utts)
        if self.training:
            # create label variable and compute loss
            labels = {s: [len(self.ontology.values[s]) * [0] for i in range(len(batch))] for s in self.ontology.slots}

            for i, e in enumerate(batch):
                for s, v in e.turn_label:
                    s = s.lower()
                    if v not in self.ontology.values[s]:
                        labels[s][i][0] = 1
                    else:
                        labels[s][i][self.ontology.values[s].index(v)] = 1
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
