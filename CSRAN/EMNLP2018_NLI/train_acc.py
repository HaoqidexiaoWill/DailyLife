#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings
warnings.filterwarnings("ignore",
        message="numpy.dtype size changed")
warnings.filterwarnings("ignore",
        message="numpy.ufunc size changed")

from collections import Counter
import csv
import argparse
from keras.preprocessing import sequence
from datetime import datetime
import numpy as np
import random
import os
from tqdm import tqdm
from utilities import *
import time
import tensorflow as tf
import sys
from sklearn.utils import shuffle
from collections import Counter
import six
if six.PY2:
    import cPickle as pickle
else:
    import pickle
import codecs
from keras.utils import np_utils
from tf_models.model import Model
import string
import re
import operator
from utilities import *
from collections import defaultdict
import sys
from nltk.corpus import stopwords
from tylib.lib.viz import *
from tensor_log import Logger
tf_logger = Logger('./tensorboard')

from tylib.exp.experiment import Experiment
from tylib.exp.exp_ops import *
from tylib.exp.tuning import *
from parser import *

reload(sys)
sys.setdefaultencoding('UTF8')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def batchify(data, i, bsz, max_sample):
    start = int(i * bsz)
    end = int(i * bsz) + bsz
    if(end>max_sample):
        end = max_sample
    data = data[start:end]
    return data

class NLIExperiment(Experiment):
    """ Implements a NLI experiment for training and evaluating models.
    """

    def __init__(self, inject_params=None):
        print("Starting NLI Experiment")
        super(NLIExperiment, self).__init__()
        self.uuid = datetime.now().strftime("%d:%m:%H:%M:%S")
        self.parser = build_parser()
        self.patience = 0

        self.char_index = {}
        self.pos_index = {}

        self.args = self.parser.parse_args()
        self.show_metrics = ['ACC','LOSS']
        self.eval_primary = 'ACC'

        if(self.args.default_len==1):
            """ ignore the abundance of datasets here
            I used this general code repo for all datasets
            (filtered for NLI dataset..)
            """
            if(self.args.dataset=='TrecQA'):
                self.args.qmax = 28
                self.args.amax = 60
            elif(self.args.dataset=='WikiQA'):
                self.args.qmax = 20
                self.args.amax = 60
            elif(self.args.dataset=='InsuranceQA'):
                import insurance_qa_data_helpers
                self.args.qmax = 10
                self.args.amax = 100
                self.args.supply_neg = True
                self.thresholds = [(0, 20), (20, 40)]
                vocab1,vocab = insurance_qa_data_helpers.insurance_read_vocab()
                id2sen = insurance_qa_data_helpers.insurance_read_sen2id(vocab1)

                raw = insurance_qa_data_helpers.insurance_read_train(vocab1,id2sen)
                random.shuffle(raw)
                self.train_set = raw
                self.dev_set = insurance_qa_data_helpers.insurance_read_test(vocab1,id2sen)
                self.dev_set_raw = self.dev_set
                self.word_index = vocab
                """
                # 词映射ID
                vocab = insurance_qa_data_helpers.build_vocab()
                # raw语料,记录所有train里的raw数据
                raw = insurance_qa_data_helpers.read_raw()
                testList, vectors = insurance_qa_data_helpers.load_test_and_vectors()
                random.shuffle(raw)
                self.train_set = raw
                self.dev_set = testList
                self.dev_set_raw = self.dev_set
                self.word_index =vocab
                self.vectors = vectors
                # self.char_index =
                """
            elif(self.args.dataset=='YahooQA'):
                import insurance_qa_data_helpers
                self.args.qmax = 20
                self.args.amax = 50
                self.args.num_neg = 5
                with codecs.open('./env.pkl', 'rb') as f:
                    env = pickle.load(f)
                self.word_index = env['word_index']
                self.train_set = insurance_qa_data_helpers.clean_train(env['train'])
                self.dev_set = insurance_qa_data_helpers.clean_train(env['dev'])
                self.dev_set_raw = self.dev_set
            elif(self.args.dataset=='Quora'):
                self.args.qmax = 50
                self.args.amax = 50
                self.thresholds = [(0,10),(10,25)]
            elif(self.args.dataset=='SNLI'):
                self.args.qmax = 80
                self.args.amax = 60
                self.args.num_class = 3
                self.thresholds = [(0,20),(21,50)]
            elif(self.args.dataset=='Ubuntu'):
                self.args.qmax = 40
                self.args.amax = 40
                self.thresholds = [(0,20),(20,40)]

        print("Setting up environment..")
        # self.env = dictFromFileUnicode(
        #     './datasets/{}/env.gz'.format(self.args.dataset))
        self.model_name = self.args.rnn_type
        self._setup()

        if(inject_params is not None):
            for param, val in inject_params.items():
                setattr(self.args, param, val)
                self.write_to_file("[Injection] {} to {}".format(
                                                param, val))
        self._load_sets()

        print("num data={}".format(len(self.train_set)))
        self.mdl = Model(self.vocab, self.args,
                        char_vocab=len(self.char_index),
                        pos_vocab=len(self.pos_index))
        self._print_model_stats()
        self.hyp_str = self.model_name + '_' + self.uuid
        self._setup_tf()

    def prepare_set(self, data, num_neg=1, pointwise=False,
                    local_feats=None, pos_feats=None, kg_feats=None):
        """ Prepares dataset for training/eval
        """
        # prepares dataset with negative sampling
        self.char_pad_token = [0 for i in range(self.args.char_max)]

        def word2id(word):
            if(word in self.word_index):
                return self.word_index[word]
            else:
                return 1

        def sent2char(sent, pad_max):
            def word2char(word):
                word = [self.char_index[x] for x in word]
                word = pad_to_max(word, self.args.char_max)
                return word

            sent_chars = [word2char(x) for x in sent]
            pad_token = [0 for i in range(self.args.char_max)]
            sent_chars = pad_to_max(sent_chars, pad_max,
                            pad_token=pad_token)
            return sent_chars

        def text2ids(txt):
            txt = [x for x in txt if len(x)>0]
            if(self.args.use_lower):
                txt = [x.lower() for x in txt]
            if(len(txt)==0):
                return [0]
            _txt = [word2id(x) for x in txt]
            return _txt, txt

        def pos2ids(txt):
            txt = [x for x in txt if len(x)>0]
            if(len(txt)==0):
                return [0]
            _txt = [self.pos_index[x] for x in txt]
            return _txt

        def char_ids(txt, sent_max):
            txt = [x for x in txt if len(x)>0]
            _txt = [[self.char_index[y] for y in x] for x in txt]
            _txt = [pad_to_max(x, self.args.char_max) for x in _txt]
            _txt = pad_to_max(_txt, sent_max,
                    pad_token=self.char_pad_token)
            return _txt

        if(self.args.use_openlm==1):
            # Use openLM vocab instead
            print("Using OpenLM vocab")
            q1 = [' '.join(x[0]) for x in data]
            q2 = [' '.join(x[1]) for x in data]
            q1 = self.TextEncoder.encode(q1)
            q2 = self.TextEncoder.encode(q2)
        else:
            o1 = [text2ids(x[0]) for x in data]
            o2 = [text2ids(x[1]) for x in data]
            q1 = [x[0] for x in o1]
            q2 = [x[0] for x in o2]
            e1 = [x[1] for x in o1]
            e2 = [x[1] for x in o2]

        q1_len = [len(x) for x in q1]
        q2_len = [len(x) for x in q2]

        label = [x[2] for x in data]
        print(Counter(label))

        print("======================================")
        print("Showing Meta Stats")
        print("Max q1=", np.max(q1_len))
        print("Max q2=", np.max(q2_len))
        print("Avg q1=", np.mean(q1_len))
        print("Avg q2=", np.mean(q2_len))
        print("Min q1=", np.min(q1_len))
        print('Min q2=', np.min(q2_len))
        if(self.args.dataset!='SICK'):
            print(Counter(label))
        print("=====================================")

        q1 = sequence.pad_sequences(q1, maxlen=self.args.qmax,
                                    padding=self.args.padding)
        q2 = sequence.pad_sequences(q2, maxlen=self.args.amax,
                                    padding=self.args.padding)

        q1_len = [min(x,self.args.qmax) for x in q1_len]
        q2_len = [min(x,self.args.amax) for x in q2_len]
        output = [q1, q1_len, q2, q2_len]

        self.mdl.register_index_map(0, 'q1_inputs')
        self.mdl.register_index_map(1, 'q1_len')
        self.mdl.register_index_map(2, 'q2_inputs')
        self.mdl.register_index_map(3, 'q2_len')

        if('CHAR' in self.args.rnn_type):
            print("Preparing Chars...")
            c1 = [char_ids(x[0], self.args.qmax) for x in tqdm(data)]
            c2 = [char_ids(x[1], self.args.amax) for x in tqdm(data)]
            c1 = np.array(c1).reshape((-1,
                    self.args.char_max * self.args.qmax))
            c2 = np.array(c2).reshape((-1,
                self.args.char_max * self.args.amax))
            print(c1.shape)
            print(c2.shape)
            self.mdl.register_index_map(len(output),
                            'c1_inputs')
            output.append(c1)
            self.mdl.register_index_map(len(output),
                            'c2_inputs')
            output.append(c2)

        if(self.args.local_feats==1 and local_feats is not None):
            print("Using Local Features..")
            f1 = [pad_to_max(x[0], self.args.qmax) for x in local_feats]
            f2 = [pad_to_max(x[1], self.args.amax) for x in local_feats]
            f1 = np.array(f1).reshape((-1, self.args.qmax, 1))
            f2 = np.array(f2).reshape((-1, self.args.amax, 1))
            self.mdl.register_index_map(len(output),
                            'f1_inputs')
            output.append(f1)
            self.mdl.register_index_map(len(output),
                            'f2_inputs')
            output.append(f2)

        if(self.args.use_pos==1 and pos_feats is not None):
            print("Using Pos Features..")
            p1 = [pos2ids(x[0]) for x in pos_feats]
            p2 = [pos2ids(x[1]) for x in pos_feats]
            p1 = [pad_to_max(x, self.args.qmax) for x in p1]
            p2 = [pad_to_max(x, self.args.amax) for x in p2]
            p1 = np.array(p1).reshape((-1, self.args.qmax))
            p2 = np.array(p2).reshape((-1, self.args.amax))
            self.mdl.register_index_map(len(output),
                            'p1_inputs')
            output.append(p1)
            self.mdl.register_index_map(len(output),
                            'p2_inputs')
            output.append(p2)

        if(self.args.use_elmo==1):
            q1_str = e1
            q1_str = [pad_to_max(x, self.args.qmax, pad_token='__pad__') for x in q1_str]
            q2_str = e2
            q2_str = [pad_to_max(x, self.args.amax, pad_token='__pad__') for x in q2_str]
            self.mdl.register_index_map(len(output),
                            'q1_elmo_inputs')
            output.append(q1_str)
            self.mdl.register_index_map(len(output),
                            'q2_elmo_inputs')
            output.append(q2_str)

        # label always last
        output.append(label)

        output = zip(*output)

        original_len = len(output)
        output = [x for x in output if x[1]>0 and x[3]>0]
        print('Filtered={}'.format(original_len-len(output)))
        return output

    def _load_sets(self):
        # Load train, test and dev sets
        # fp = './datasets/fold{}/'.format(self.args.fold)
        #self.train_set = self.env['train']
        #self.dev_set = self.env['dev']
        if(self.args.dev==0):
            self.train_set += self.dev_set
        #self.test_set = self.env['test']

        if('CHAR' in self.args.rnn_type):
            self.char_index = self.env['char_index']

        if(self.args.local_feats==1):
            # Use exact match features
            print("Using Local Feats [EM]")
            self.feat_env =dictFromFileUnicode(
                        './datasets/{}/feat_env.gz'.format(
                            self.args.dataset))
            self.train_feats = self.feat_env['train_feats']
            self.dev_feats = self.feat_env['dev_feats']
            self.test_feats = self.feat_env['test_feats']
            if(self.args.dataset=='MNLI'):
                self.dev2_feats = self.feat_env['dev2_feats']
                self.test2_feats = self.feat_env['test2_feats']
                self.train2_feats = self.feat_env['train2_feats']
        else:
            self.train_feats = None
            self.train2_feats = None
            self.dev_feats = None
            self.test_feats = None
            self.test2_feats = None
            self.dev2_feats = None

        # Experimental KG features (not used in the end)
        self.train_kg = None
        self.train2_kg = None
        self.dev_kg = None
        self.test_kg = None
        self.test2_kg = None
        self.dev2_kg = None

        if(self.args.use_pos==1):
            # Use Pos tag features
            print("Using POS tag features [POS]")
            self.pos_env =dictFromFileUnicode(
                        './datasets/{}/pos_env.gz'.format(
                            self.args.dataset))
            self.train_pos = self.pos_env['train_pos']
            self.dev_pos = self.pos_env['dev_pos']
            self.test_pos = self.pos_env['test_pos']
            self.pos_index = self.env['pos_index']
            # print(self.dev_pos[:10])
            if(self.args.dataset=='MNLI'):
                self.dev2_pos = self.pos_env['dev2_pos']
                self.test2_pos = self.pos_env['test2_pos']
                self.train2_pos = self.pos_env['train2_pos']
        else:
            self.train_pos = None
            self.train2_pos = None
            self.dev_pos = None
            self.test_pos = None
            self.test2_pos = None
            self.dev2_pos = None

        #self.word_index = self.env['word_index']
        self.index_word = {k:v for v, k in self.word_index.items()}
        # self.vocab = len(self.word_index)
        self.vocab = len(self.word_index)
        #  print(self.env.keys())
        self.predict_dict = None
        self.test_predict_dict = defaultdict(int)
        print("vocab={}".format(self.vocab))
        #if(self.args.features and 'word2dfs' in self.env):
        if False:
            word2df = self.env['word2dfs']
            id2df = {}
            for key, value in word2df.items():
                _id = self.word_index[key]
                id2df[_id] = value
            self.word2df = id2df
            print("Loaded word2dfs")
        else:
            self.word2df = None

        self.stoplist = set(stopwords.words('english'))
        if(self.args.supply_neg):
            self.answer_pool = []
            for sample in self.train_set:
                self.answer_pool.append(sample[1])
            print("Built answer pool of {} answers".format(
                                    len(self.answer_pool)))
        self.write_to_file("Train={} Dev={} | questions".format(
                len(self.train_set),len(self.dev_set)))

    def _print_model_output(self, raw_preds, set_type='', name=''):
        """ Prints model output
        (For ensemblining etc..)
        """
        fp = self.out_dir +'./{}_{}.txt'.format(set_type, name)
        with open(fp, 'w+') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(raw_preds)

    def make_submission(self, data, bsz, epoch, name="", set_type="",
                        meta=None, dev_score=None):
        """ Make submission for MNLI
        """
        fp = self.out_dir +'./{}.txt'.format(set_type)
        num_batches = int(len(data) / bsz)
        all_preds = []
        raw_preds = []
        for i in tqdm(range(num_batches+1)):
            batch = batchify(data, i, bsz, max_sample=len(data))
            if(len(batch)==0):
                continue
            feed_dict = self.mdl.get_feed_dict(batch, mode='testing')
            p, preds = self.sess.run([self.mdl.predict_op,
                                self.mdl.predictions], feed_dict)
            all_preds += preds.tolist()
            raw_preds += p.tolist()

        convertor = ['contradiction','neutral', 'entailment']

        pair_id = [x[0] for x in meta]
        result = [convertor[x] for x in all_preds]
        write_data = zip(pair_id, result)
        with open(fp, 'w+') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['pairID','gold_label'])
            writer.writerows(write_data)
        meta_file = self.out_dir +'./meta_{}.txt'.format(set_type)
        with open(meta_file, 'w+') as f:
            f.write(str(dev_score) + ',' + str(epoch))
        return all_preds, raw_preds

    def qualitative_output(self, feats, data):
        """ For viz purposes only
        """
        feats = zip(*feats)
        output = []
        for i, d in enumerate(data):
            # print("=====================================")
            _feats = feats[i]
            # print(d)
            _q1 = [self.index_word[x] for x in d[0] if x!=0]
            _q2 = [self.index_word[x] for x in d[2] if x!=0]
            q1 = ' '.join(_q1)
            q2 = ' '.join(_q2)
            _output = [q1, q2]
            for i in range(6):
                o1 = ' ' .join([str(x[i]) for x in _feats[0][:len(_q1)]])
                o2 = ' '.join([str(x[i]) for x in _feats[1][:len(_q2)]])
                _output.append(o1)
                _output.append(o2)
            _output.append(d[-1])
            output.append(_output)
        return output

    def evaluate(self, data, bsz, epoch, name="", set_type=""):
        acc = 0
        num_batches = int(len(data) / bsz)
        all_preds = []
        raw_preds = []
        ff_feats = []
        all_qout = []
        actual_labels = [x[-1] for x in data]
        losss = []
        for i in tqdm(range(num_batches+1)):
            batch = batchify(data, i, bsz, max_sample=len(data))
            if(len(batch)==0):
                continue
            feed_dict = self.mdl.get_feed_dict(batch, mode='testing')
            a, p, preds,loss = self.sess.run([self.mdl.accuracy,
                            self.mdl.predict_op,
                            self.mdl.predictions,
                            self.mdl.cost], feed_dict)
            all_preds += preds.tolist()
            raw_preds += p.tolist()
            acc += (a * len(batch))
            losss.append(loss)
        qout_dir = './{}_{}_out.txt'.format(self.args.dataset, set_type)
        with open(qout_dir, 'w+') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(all_qout)
        acc = acc / len(data)
        print(Counter(all_preds))
        self._register_eval_score(epoch, set_type, 'ACC', acc)

        raw_preds = [[x[0],x[1], actual_labels[i]]
                        for i, x in enumerate(raw_preds)]
        return acc, raw_preds,losss

    def train(self):
        """ Function to train and evaluate models
        """
        scores = []
        best_score = -1
        best_dev = -1
        best_epoch = -1
        counter = 0
        epoch_scores = {}
        self.eval_list = []
        data = self.prepare_set(self.train_set, num_neg=0,
                                pointwise=True,
                                local_feats=self.train_feats,
                                pos_feats=self.train_pos,
                                kg_feats=self.train_kg
                                )
        self.dev_set = self.prepare_set(self.dev_set, num_neg=0,
                                pointwise=True,
                                local_feats=self.dev_feats,
                                pos_feats=self.dev_pos,
                                kg_feats=self.dev_kg
                                )
        if(self.args.sort_batch==1 and self.args.use_snli==0):
            # sort samples
            """ Sort batch into buckets
            """
            print("Using sort_batch mode with threshold")
            print(self.thresholds)
            train_data = optimize_batch(data, thresholds=self.thresholds)
            if(self.args.use_snli==1):
                train_data2 = optimize_batch(self.train_set2,
                                        thresholds=self.thresholds)

        else:
            if(self.args.use_snli==1):
                train_data2 = self.train_set2
            train_data = data
        lr = self.args.learn_rate
        for epoch in range(1, self.args.epochs+1):
            all_att_dict = {}
            pos_val, neg_val = [],[]
            t0 = time.clock()
            self.write_to_file("=====================================")
            losses = []

            if(self.args.sort_batch==1):
                data = optimized_batch_shuffle(train_data)
            else:
                random.shuffle(train_data)
                data = train_data

            num_batches = int(len(data) / self.args.batch_size)
            norms = []
            all_acc = []
            self.sess.run(tf.assign(self.mdl.is_train, self.mdl.true))
            preddd = []
            for i in tqdm(range(0, num_batches+1)):
                """ Main training loop
                """
                batch = batchify(data, i, self.args.batch_size,
                                max_sample=len(data))
                if(len(batch)==0):
                    continue
                feed_dict = self.mdl.get_feed_dict(batch, lr=lr)
                train_op = self.mdl.train_op
                _, loss, preds,a = self.sess.run([train_op,
                                            self.mdl.cost,
                                            self.mdl.predict_op,
                                            self.mdl.accuracy],
                                            feed_dict)
                all_acc.append(a)
                counter +=1
                tf_logger.scalar_summary('train_loss', loss, (epoch - 1) * num_batches + i)
                if (i % 1000 == 0):
                    self.write_to_file("[{}] [Epoch {}{}] [{}] loss={} acc={}".format(
                        self.args.dataset, epoch, i,self.model_name,
                        loss, np.mean(all_acc)))

                    self.sess.run(tf.assign(self.mdl.is_train, self.mdl.false))
                    _, dev_preds,dev_loss = self.evaluate(self.dev_set,
                                                 self.args.batch_size, epoch, set_type='Dev')
                    tf_logger.scalar_summary('dev_loss', np.mean(dev_loss), (epoch - 1) * num_batches + i)
                    preds = [x[1] for x in dev_preds]
                    labels = [x[-1] for x in dev_preds]
                    print(preds[:5])
                    f1,thred = self.f1_score(preds,labels)
                    self.write_to_file("f1 {} thred {}".format(f1,thred))
                    tf_logger.scalar_summary('dev_f1', f1, (epoch - 1) * num_batches + i)
                    self.dev_step(preds, epoch,i,num_batches)
                    losses=[]
                    self.sess.run(tf.assign(self.mdl.is_train, self.mdl.true))
            # print("Max Norm={}".format(np.mean(norms)))

                """
                self._show_metrics(epoch, self.eval_dev,
                                    self.show_metrics,
                                        name='Dev')

                best_epoch1, cur_dev = self._select_test_by_dev(epoch,
                                                        self.eval_dev,
                                                        {},
                                                        no_test=True)
                
                if(self.args.dev_lr>0 and best_epoch1!=epoch):
                    self.patience +=1
                    print('Patience={}'.format(self.patience))
                    if(self.patience>=self.args.patience):
                        print("Reducing LR by {} times".format(self.args.dev_lr))
                        lr = lr / self.args.dev_lr
                        print("LR={}".format(lr))
                        self.patience = 0
                """
    def dev_step(self,dev_preds,epoch,i,num_batches):
        scoreList = list()
        sessdict = {}
        index = 0
        for line in self.dev_set_raw:
            qid = str(line[0])
            if not qid in sessdict:
                sessdict[qid] = list()
            sessdict[qid].append((dev_preds[index], line[2]))
            index += 1
            if index >= len(self.dev_set):
                break
        lev1 = .0
        lev0 = .0
        for k, v in sessdict.items():
            v.sort(key=operator.itemgetter(0), reverse=True)
            score, flag = v[0]
            if flag == 1:
                lev1 += 1
            if flag == 0:
                lev0 += 1
        # 回答的正确数和错误数
        print ('回答正确数 ' + str(lev1))
        print ('回答错误数 ' + str(lev0))
        print ('准确率 ' + str(float(lev1)/(lev1+lev0)))
        tf_logger.scalar_summary('dev_acc', float(lev1)/(lev1+lev0), (epoch-1)*num_batches+i)

    def f1_score(self,preds, labels):
        assert len(preds) == len(labels), 'F1预测和标签长度不一致'
        mxf = 0.0
        thred = 0.0
        for i in range(100):
            fk = i / 99.0
            f1 = 0.0
            f1_sum = 1.0
            f2_sum = 1.0
            for i in range(len(preds)):
                if preds[i] > fk:
                    f1_sum += 1
                if labels[i] == 1:
                    f2_sum += 1
                if preds[i] > fk and labels[i] == 1:
                    f1 += 1
            evc = 2 * f1 / (f1_sum + f2_sum)
            if evc > mxf:
                thred = fk
            mxf = max(evc, mxf)
        return mxf, thred

if __name__ == '__main__':
    exp = NLIExperiment(inject_params=None)
    exp.train()
    print("End of code!")
