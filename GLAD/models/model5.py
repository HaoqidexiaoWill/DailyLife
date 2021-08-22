import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import logging
import os
import re
import json
from collections import defaultdict
from pprint import pformat

from models.TCN import TemporalConvNet
#from models.Pos import PositionalEncoder
# import ipdb



def position_encoding_init(n_position, emb_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])

    # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    # apply cos on 1st,3rd,5th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def pad(seqs, emb, device, pad=0):
    lens = [len(s) for s in seqs]
    max_len = max(lens)
    padded = torch.LongTensor([s + (max_len-l) * [pad] for s, l in zip(seqs, lens)])
    #这个时候padded已经是个张量了 torch.size(7,5)
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
def run_cnn_1(self,inputs):

    outputs = inputs.unsqueeze(1)
    # outputs = self.conv1(outputs)       #torch.Size([50, 16, 30, 400])
    # outputs = self.conv2(outputs)       #torch.Size([50, 128, 30, 400])
    # outputs = self.conv3(outputs)       #torch.Size([50, 256, 30, 400])
    # outputs = self.conv4(outputs)       #torch.Size([50, 400, 30, 1])

    outputs = self.conv_final(outputs)       #torch.Size([50, 128, 30, 400])
    # outputs = self.convNN(outputs,3)
    # print(outputs.size())
    # exit()
    outputs = outputs.squeeze(3).transpose(1, 2)

    return outputs+inputs

def run_cnn_2(self,inputs):
    outputs = inputs.unsqueeze(1)
    # outputs = self.conv1(outputs)       #torch.Size([50, 16, 30, 400])
    # outputs = self.conv2(outputs)       #torch.Size([50, 128, 30, 400])
    # outputs = self.conv3(outputs)       #torch.Size([50, 256, 30, 400])
    # outputs = self.conv4(outputs)       #torch.Size([50, 400, 30, 1])

    outputs = self.conv_final(outputs)       #torch.Size([50, 128, 30, 400])
    # outputs = self.convNN(outputs,3)
    # print(outputs.size())
    # exit()
    outputs = outputs.squeeze(3).transpose(1, 2)
    outputs = outputs+inputs

    outputs_ = outputs.unsqueeze(1)
    outputs_ = self.conv_final(outputs_)       #torch.Size([50, 128, 30, 400])
    outputs_ = outputs_.squeeze(3).transpose(1, 2)

    return outputs+outputs_

def run_cnn_3(self,inputs):
    #inputs = inputs.unsqueeze(1)
    # outputs = self.conv1(outputs)       #torch.Size([50, 16, 30, 400])
    # outputs = self.conv2(outputs)       #torch.Size([50, 128, 30, 400])
    # outputs = self.conv3(outputs)       #torch.Size([50, 256, 30, 400])
    # outputs = self.conv4(outputs)       #torch.Size([50, 400, 30, 1])

    outputs = inputs

    for i in range(3):
        outputs_ = self.conv_final(outputs.unsqueeze(1))             #torch.Size([50, 400, 30, 1])
        outputs_ = outputs_.squeeze(3).transpose(1, 2)  #torch.Size([50,30, 400])
        outputs = outputs.squeeze(2)
        outputs = outputs+outputs_

    return outputs


def run_cnn_4(self,inputs):

    outputs = inputs.unsqueeze(1)
    # outputs = self.conv1(outputs)       #torch.Size([50, 16, 30, 400])
    # outputs = self.conv2(outputs)       #torch.Size([50, 128, 30, 400])
    # outputs = self.conv3(outputs)       #torch.Size([50, 256, 30, 400])
    # outputs = self.conv4(outputs)       #torch.Size([50, 400, 30, 1])

    outputs = self.conv_slot(outputs)       #torch.Size([50, 128, 30, 400])
    # outputs = self.convNN(outputs,3)

    outputs = outputs.squeeze(3).transpose(1, 2)

    outputs.contiguous().view(-1, outputs.size(2))
    outputs = self.mapping(outputs)
    A, B = outputs.split(2*self.hidden_size, 2) # A, B: batch, seq_len, hidden
    A2 = A.contiguous().view(-1, A.size(2)) # A2: batch * seq_len, hidden
    B2 = B.contiguous().view(-1, B.size(2)) # B2: batch * seq_len, hidden
    attn = torch.mul(A2, self.softmax(B2)) # attn: batch * seq_len, hidden
    outputs = attn.view(A.size(0), A.size(1), -1) # outputs: batch, seq_len, 2 * hidden
    return outputs+inputs

def run_cnn_5(self,inputs):
    outputs = inputs.unsqueeze(1)

    # print(self.conv_slot2)
    # exit()

    outputs = self.conv_slot2(outputs)       #torch.Size([50, 128, 30, 400])
    outputs = outputs.squeeze(3).transpose(1, 2)

    return outputs+inputs

def run_cnn_6(self,inputs):
    outputs = inputs.unsqueeze(1)
    outputs = self.conv_final(outputs)       #torch.Size([50, 128, 30, 400])
    outputs = outputs.squeeze(3).transpose(1, 2)
    outputs.contiguous().view(-1, outputs.size(2))
    outputs = self.mapping(outputs)
    A, B = outputs.split(2*self.hidden_size, 2) # A, B: batch, seq_len, hidden
    A2 = A.contiguous().view(-1, A.size(2)) # A2: batch * seq_len, hidden
    B2 = B.contiguous().view(-1, B.size(2)) # B2: batch * seq_len, hidden
    attn = torch.mul(A2, self.softmax(B2)) # attn: batch * seq_len, hidden
    outputs = attn.view(A.size(0), A.size(1), -1) # outputs: batch, seq_len, 2 * hidden  
    outputs = outputs+inputs

    outputs_ = outputs.unsqueeze(1)
    outputs_ = self.conv_final(outputs_)       #torch.Size([50, 128, 30, 400])
    outputs_ = outputs_.squeeze(3).transpose(1, 2)
    outputs.contiguous().view(-1, outputs.size(2))
    outputs = self.mapping(outputs)
    A, B = outputs.split(2*self.hidden_size, 2) # A, B: batch, seq_len, hidden
    A2 = A.contiguous().view(-1, A.size(2)) # A2: batch * seq_len, hidden
    B2 = B.contiguous().view(-1, B.size(2)) # B2: batch * seq_len, hidden
    attn = torch.mul(A2, self.softmax(B2)) # attn: batch * seq_len, hidden
    outputs = attn.view(A.size(0), A.size(1), -1) # outputs: batch, seq_len, 2 * hidden

    return outputs+outputs_
def run_cnn_7(self,inputs):

    inputs_ = self.mapping(inputs)
    A,B = inputs_.split(2*self.hidden_size, 2)
    A_inputs = A.unsqueeze(1)
    A_outputs =self.conv_slot(A_inputs)
    A_outputs = A_outputs.squeeze(3).transpose(1, 2)
    B_inputs = B.unsqueeze(1)
    B_outputs =self.conv_slot2(B_inputs)
    B_outputs = B_outputs.squeeze(3).transpose(1, 2)


    A2 = A_outputs.contiguous().view(-1, A_outputs.size(2))
    B2 = B_outputs.contiguous().view(-1, B_outputs.size(2))

    attn = torch.mul(A2, self.softmax(B2)) # attn: batch * seq_len, hidden
    outputs = attn.view(A.size(0), A.size(1), -1) # outputs: batch, seq_len, 2 * hidden
    return outputs+inputs
def run_cnn(self,inputs):

    inputs_ = self.mapping(inputs)
    A,B = inputs_.split(2*self.hidden_size, 2)
    A_inputs = A.unsqueeze(1)
    A_outputs =self.conv_slot(A_inputs)
    A_outputs = A_outputs.squeeze(3).transpose(1, 2)
    B_inputs = B.unsqueeze(1)
    B_outputs =self.conv_slot2(B_inputs)
    B_outputs = B_outputs.squeeze(3).transpose(1, 2)


    A2 = A_outputs.contiguous().view(-1, A_outputs.size(2))
    B2 = B_outputs.contiguous().view(-1, B_outputs.size(2))

    attn = torch.mul(A2, self.softmax(B2)) # attn: batch * seq_len, hidden
    outputs_ = attn.view(A.size(0), A.size(1), -1) # outputs: batch, seq_len, 2 * hidden
    outputs_ = outputs_+inputs

    outputs2 = self.mapping(outputs_)
    A,B = outputs2.split(2*self.hidden_size,2)
    A_inputs = A.unsqueeze(1)
    A_outputs =self.conv_slot(A_inputs)
    A_outputs = A_outputs.squeeze(3).transpose(1, 2)
    B_inputs = B.unsqueeze(1)
    B_outputs =self.conv_slot2(B_inputs)
    B_outputs = B_outputs.squeeze(3).transpose(1, 2)

    A2 = A_outputs.contiguous().view(-1, A_outputs.size(2))
    B2 = B_outputs.contiguous().view(-1, B_outputs.size(2))

    attn = torch.mul(A2, self.softmax(B2)) # attn: batch * seq_len, hidden
    outputs = attn.view(A.size(0), A.size(1), -1) # outputs: batch, seq_len, 2 * hidden


    return outputs+outputs_


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
        # print(out.size())
        out.detach_()
        return F.dropout(out, self.dropout, self.training)
class PositionalEmbedding(nn.Embedding):
    """
    Sets up embedding layer for word sequences as well as for word positions.Both the layers are trainable.
    Returns embeddings of words which also contains the position(time) component
    """
    def __init__(self, *args, dropout=0, **kwargs):    
        """
        Args:
            vocab_size  : [int] vocabulary size
            emb_dim     : [int] embedding dimension for words
            max_len     : [int] maxlen of input sentence
            batch_size  : [int] batch_size

        Returns:
            position encoded word embeddings

        Raises:
            nothing
        """   
        super(PositionalEmbedding,self).__init__(*args, **kwargs)
        #{'num_embeddings': 954, 'embedding_dim': 400}  ()
        #super().__init__(*args, **kwargs)

        self.batch_size = 50
        self.emb_dim = kwargs['embedding_dim']
        self.max_len = 50

        n_position = self.max_len+1
        self.position_enc = torch.nn.Embedding(n_position, self.emb_dim, padding_idx=0)
        self.position_enc.weight.data = position_encoding_init(n_position, self.emb_dim)
        
    def get_absolute_pos(self,word_sequences):
        batch = []
        for word_seq in word_sequences:
            start_idx = 1
            word_pos = []
            for pos in word_seq:
                if int(pos) == 0:
                    word_pos.append(0)
                else:
                    word_pos.append(start_idx)
                    start_idx+=1
            batch.append(torch.from_numpy(np.array(word_pos)).type(torch.LongTensor))

        #batch = torch.cat(batch)        #torch.Size([35])
        
        # batch = torch.cat(batch).view(self.batch_size,self.max_len)   
        batch = torch.cat(batch).view(word_sequences.size(0),word_sequences.size(1))        #torch.Size([7, 5]) 1  
        return Variable(batch)
        
    def forward(self,*args, **kwargs):          #args 是一个张量元组
        word_embeddings = super().forward(*args, **kwargs)  #输入是torch.Size([7, 5]) 输出是torch.Size([7, 5, 400])
        # print(word_embeddings.size())
        # exit()
        word_pos = self.get_absolute_pos(*args, **kwargs)

        word_pos_encoded = word_embeddings + self.position_enc(word_pos.cuda())
        return word_pos_encoded

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

        self.hidden_size = int(200)
        self.kernel_size = (1,400)
        self.kernel_size2 = (3,400)
        self.kernel_sizes = [1]
        self.in_channels = 1
        # self.out_channels = self.hidden_size * 2
        self.out_channels = 400
        self.padding = (1//2, 0)
        self.stride = 1

        self.conv = nn.Conv2d(self.in_channels, self.out_channels,self.kernel_size, self.stride, self.padding)
        # self.convs = [nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 400), stride=(1, 1), padding=(K//2, 0)) for K in self.kernel_sizes]
        self.convs = [nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, (K//2, 0)) for K in self.kernel_sizes]

        self.batch_norm = nn.BatchNorm2d(self.out_channels)


        self.affine = nn.Linear(4*self.hidden_size, 2*self.hidden_size)
        self.layers = 1

        self.softmax = nn.Softmax()
        self.mapping = nn.Linear(2*self.hidden_size, 4* self.hidden_size)

        self.conv_single = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=400, kernel_size=self.kernel_size,stride=1, padding=self.padding),nn.ReLU(),)
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1,stride=1, padding=self.padding),nn.ReLU(),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1,stride=1, padding=self.padding),nn.ReLU(),)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1,stride=1, padding=self.padding),nn.ReLU(),)
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=400, kernel_size=self.kernel_size,stride=1, padding=self.padding),nn.ReLU(),)
        self.conv_final = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=400, kernel_size=self.kernel_size,stride=1, padding=self.padding),nn.ReLU(),)

        self.conv_slot = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=400, kernel_size=self.kernel_size,stride=1, padding=self.padding),nn.ReLU(),)
        self.conv_slot2 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=400, kernel_size=self.kernel_size2,stride=1, padding=(3//2,0)),nn.ReLU(),)


        self.convNN = nn.Linear(32, 400)

        # self.tcn = TemporalConvNet(
        #     400, 
        #     [self.out_channels]*2, 
        #     self.kernel_size, dropout=0.2, max_length=30, attention='attention')



 
    def beta(self, slot):
        return F.sigmoid(self.beta_raw[self.slots.index(slot)])

    def forward(self, x, x_len, slot, slot_emb, default_dropout=0.2):
        beta = self.beta(slot)
        x_new = torch.cat((slot_emb.unsqueeze(0).expand_as(x), x), dim=2)
        #global_h = run_rnn(self.global_rnn, x_new, x_len)
        global_h = run_cnn(self, x)
        #global_h = self.tcn(x.transpose(1,2)).transpose(1,2)
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
        #self.emb_fixed_old = FixedEmbedding(len(vocab), args.demb, dropout=args.dropout.get('emb', 0.2))
        self.emb_fixed = FixedEmbedding(len(vocab), args.demb, dropout=args.dropout.get('emb', 0.2))

        #self.emb_fixed = PositionalEmbedding(nu m_embeddings = len(vocab),embedding_dim = args.demb, dropout=args.dropout.get('emb', 0.2))
        self.encoder = GCEencoder

        self.utt_encoder = self.encoder(args.demb, args.dhid, self.ontology.slots, dropout=args.dropout)
        self.act_encoder = self.encoder(args.demb, args.dhid, self.ontology.slots, dropout=args.dropout)
        self.ont_encoder = self.encoder(args.demb, args.dhid, self.ontology.slots, dropout=args.dropout)
        self.utt_scorer = nn.Linear(2 * args.dhid, 1)
        self.score_weight = nn.Parameter(torch.Tensor([0.5]))

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
        utterance, utterance_len = pad([e.num['transcript'] for e in batch], self.emb_fixed, self.device, pad=eos)
        acts = [pad(e.num['system_acts'], self.emb_fixed, self.device, pad=eos) for e in batch]
        ys = {}
        for s in self.ontology.slots:
            # for each slot, compute the scores for each value
            
            s_words = s.split()
            s_new = s_words[0]
            s_emb = self.emb_fixed(torch.cuda.LongTensor([self.vocab.word2index(s_new)]))
            H_utt, c_utt = self.utt_encoder(utterance, utterance_len, slot=s, slot_emb=s_emb)
            _, C_acts = list(zip(*[self.act_encoder(a, a_len, slot=s, slot_emb=s_emb) for a, a_len in acts]))
            _, C_vals = self.ont_encoder(ontology[s][0], ontology[s][1], slot=s, slot_emb=s_emb)

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

            # combine the scores
            ys[s] = F.sigmoid(y_utts + self.score_weight * y_acts)

        if self.training:
            # create label variable and compute loss
            labels = {s: [len(self.ontology.values[s]) * [0] for i in range(len(batch))] for s in self.ontology.slots}
            for i, e in enumerate(batch):
                for s, v in e.turn_label:
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
