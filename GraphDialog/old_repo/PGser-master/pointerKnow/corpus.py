import torch
from torch.utils.data import DataLoader
import json 
import os
import re
import csv
import pandas as pd
import random
import config
from data_utils import normalizeString,inputVar,outputVar,encVar

# Default word tokens
PAD_token = config.PAD_token  # Used for padding short sentences
SOS_token = config.SOS_token  # Start-of-sentence token
EOS_token = config.EOS_token  # End-of-sentence token
UNK_token = config.UNK_token  # Unkonw token


class Voc:
    def __init__(self):
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token:"UNK"}
        self.num_words = 4  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {"PAD": PAD_token, "SOS": SOS_token, "EOS": EOS_token, "UNK":UNK_token}
        self.word2count = {"UNK": 0}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token:"UNK"}
        self.num_words = 4 # Count default tokens

        for word in keep_words:
            self.addWord(word)

class Corpus(object):
    def __init__(self,conv_path,doc_path,save_dir,min_freq=2,max_history_len=200,max_response_len=30,
                      embed_file=None):
        self.conv_path = conv_path
        self.doc_path = doc_path
        self.save_dir = save_dir
        self.min_freq = min_freq
        self.max_history_len = max_history_len
        self.max_response_len = max_response_len

        prepared_data_file = "data_" + str(min_freq) + ".pt"
        prepared_vocab_file = 'vocab_' + str(min_freq) + '.pt'

        self.prepared_data_file = os.path.join(save_dir, prepared_data_file)
        self.prepared_vocab_file = os.path.join(save_dir, prepared_vocab_file)

        self.data = None
        self.read_wiki_docs()

    def load(self):
        if not (os.path.exists(self.prepared_data_file) and 
            os.path.exists(self.prepared_vocab_file)):
            self.build()
        self.load_data(self.prepared_data_file)
        self.load_vocab(self.prepared_vocab_file)

    def load_data(self, prepared_data_file=None):
        """
        load_data
        """
        prepared_data_file = prepared_data_file or self.prepared_data_file
        print("Loading prepared data from {} ...".format(prepared_data_file))
        self.data = torch.load(prepared_data_file)
        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))
    
    def load_vocab(self,prepared_vocab_file=None):
        prepared_vocab_file = prepared_vocab_file or self.prepared_vocab_file
        print("Loading prepared vocab from {} ...".format(prepared_vocab_file))
        self.vocab = torch.load(prepared_vocab_file)
        print("Vocabulary size:",self.vocab.num_words)

    def build(self):
        """
        build
        """
        print("Start to build corpus!")

        print("Reading data ...")
        train_raw = self.read_data(data_type="train")
        valid_raw = self.read_data(data_type="valid")
        test_raw = self.read_data(data_type="test")
        self.build_vocab(train_raw)

#         train_data = self.trimRareWords(train_raw)
#         valid_data = self.trimRareWords(valid_raw)
#         test_data = self.trimRareWords(test_raw)

        self.data = {"train": train_raw,
                "valid": valid_raw,
                "test": test_raw}

        print("Saving prepared data ...")
        torch.save(self.data, self.prepared_data_file)
        print("Saved prepared data to '{}'".format(self.prepared_data_file))

        print("Saving vocab ...")
        torch.save(self.vocab, self.prepared_vocab_file)
        print("Saved vocab to '{}'".format(self.prepared_vocab_file))

    def filter_pair(self, pair):
        return type(pair['src']) == str and 1 <= len(pair['src'].split()) <= self.max_history_len and type(pair['tgt']) == str and 1 <= len(pair['tgt'].split()) <= self.max_response_len

    def read_wiki_docs(self):
        path = self.doc_path
        wiki_docs = [0]*30
        for file in os.listdir(path):
            wiki_file = open(os.path.join(path,file))
            wiki_data = json.load(wiki_file)
            wiki_docs[wiki_data['wikiDocumentIdx']] = wiki_data

        self.wiki_strings = []
        for i in range(30):
            doc = []
            for j in range(4):
                doc.append(normalizeString(str(wiki_docs[i][str(j)])))
            self.wiki_strings.append(doc)

    def build_vocab(self,data):
        self.vocab = Voc()
        for pair in data:
            self.vocab.addSentence(pair['src'])
            self.vocab.addSentence(pair['tgt'])
        
        for wiki_doc in self.wiki_strings:
            for s in wiki_doc:
                self.vocab.addSentence(s)
                
        print("Counted words:", self.vocab.num_words)
        # Trim words used under the MIN_COUNT from the voc
        self.vocab.trim(self.min_freq)

    def trimRareWords(self,data):
        try:
            # Filter out pairs with trimmed words
            for index,pair in enumerate(data):
                # Check input sentence
                for key in pair:
                    for word in data[index][key].split(' '):
                        if word not in self.vocab.word2index:
                            data[index][key] = re.sub(" "+word+" "," UNK ",data[index][key])
                            data[index][key] = re.sub("^"+word+" ","UNK ",data[index][key])
                            data[index][key] = re.sub(" "+word+"$"," UNK",data[index][key])
                            data[index][key] = re.sub("^"+word+"$","UNK",data[index][key])
            return data 
        except Exception as e:
            print(pair)
            raise e

    
    def read_data(self,data_type="train"):
        data = []
        dataPath = os.path.join(self.conv_path,data_type)
        for file in os.listdir(dataPath):
            df = pd.read_csv(os.path.join(dataPath,file),sep='\t',encoding='utf-8')

            for i in df.index:
                pair = {}
                print(df.iloc[i].wikiIdx)
                print(df.iloc[i].history)
                print(df.iloc[i].response)
                pair['doc']= [self.wiki_strings[df.iloc[i].wikiIdx][docIdx] for docIdx in range(3)]
                if i > 0 and df.iloc[i].uid != df.iloc[i-1].uid:
                    pair['src'] = df.iloc[i].history
                    pair['tgt'] = df.iloc[i].response
                    data.append(pair)

        filtered_num = len(data)
        data = [pair for pair in data if self.filter_pair(pair)]
        filtered_num -= len(data)
        print(
            "Read {} {} examples ({} filtered)".format(len(data), data_type.upper(), filtered_num))
        return data

    def build_examples(self, data):
        # Returns all items for a given batch of pairs
        try:
            data.sort(key=lambda x: len(x['src'].split(" ")), reverse=True)
            doc_batch, input_batch, output_batch = [], [], []
            for pair in data:
                doc_batch.append(pair['doc'])
                input_batch.append(pair['src'])
                output_batch.append(pair['tgt'])
            doc_inp, doc_lengths = encVar(doc_batch, self.vocab)
            inp, lengths, batch_oovs, max_oov_length,extend_inp = inputVar(input_batch, self.vocab)
            output, mask, max_target_len, extend_output = outputVar(output_batch, self.vocab, batch_oovs)
            return doc_inp,doc_lengths,inp,lengths,batch_oovs,max_oov_length,extend_inp,output, mask, max_target_len,extend_output
        except Exception as e:
            print(pair)
            raise e

    def create_batches(self, batch_size, data_type="train"):
        """
        create_batches
        """
        try:
            data = self.data[data_type]
            loader = DataLoader(dataset=data,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=self.build_examples,
                                pin_memory=False)
            return loader 
        except KeyError:
            raise KeyError("Unsported data type: {}!".format(data_type))
                       