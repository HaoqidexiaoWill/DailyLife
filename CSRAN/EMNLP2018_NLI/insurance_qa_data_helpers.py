from __future__ import print_function
import numpy as np
import random

empty_vector = []
for i in range(0, 100):
    empty_vector.append(float(0.0))
onevector = []
for i in range(0, 10):
    onevector.append(float(1))
zerovector = []
for i in range(0, 10):
    zerovector.append(float(0))

def build_vocab():
    code = int(0)
    vocab = {}
    vocab['UNKNOWN'] = code
    code += 1
    for line in open('datasets/insuranceQA_v2/train'):
        items = line.strip().split(' ')
        for i in range(2, 4):
            words = items[i].split('_')
            for word in words:
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    for line in open('datasets/insuranceQA_v2/test1'):
        items = line.strip().split(' ')
        for i in range(2, 4):
            words = items[i].split('_')
            for word in words:
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    return vocab

def rand_qa(qalist):
    index = random.randint(0, len(qalist) - 1)
    return qalist[index]

def read_alist():
    alist = []
    for line in open('datasets/insuranceQA_v2/train'):
        items = line.strip().split(' ')
        alist.append(items[3])
    print('read_alist done ......')
    return alist

def vocab_plus_overlap(vectors, sent, over, size):
    global onevector
    global zerovector
    oldict = {}
    words = over.split('_')
    if len(words) < size:
        size = len(words)
    for i in range(0, size):
        if words[i] == '<a>':
            continue
        oldict[words[i]] = '#'
    matrix = []
    words = sent.split('_')
    if len(words) < size:
        size = len(words)
    for i in range(0, size):
        vec = read_vector(vectors, words[i])
        newvec = vec.copy()
        #if words[i] in oldict:
        #    newvec += onevector
        #else:
        #    newvec += zerovector
        matrix.append(newvec)
    return matrix

def load_vectors():
    # vectors = {}
    vectors = []
    for line in open('datasets/insuranceQA_v2/vectors.nobin'):
        items = line.strip().split(' ')
        if (len(items) < 101):
            continue
        vec = []
        for i in range(1, 101):
            vec.append(float(items[i]))
        # vectors[items[0]] = vec
        vectors.append(vec)
    return np.array(vectors)

def read_vector(vectors, word):
    global empty_vector
    if word in vectors:
        return vectors[word]
    else:
        return empty_vector
        #return vectors['</s>']

def load_test_and_vectors():
    raw = []
    for line in open('datasets/insuranceQA_v2/test1'):
        items = line.strip().split(' ')
        query = items[2].split('_')
        if '<a>' in query:
            query = query[:query.index('<a>')]
        answer = items[3].split('_')
        if '<a>' in answer:
            answer = answer[:answer.index('<a>')]
        raw.append([query, answer, items[0]])
    vectors = load_vectors()
    return raw, vectors

def load_train_and_vectors():
    trainList = []
    for line in open('datasets/insuranceQA_v2/train'):
        trainList.append(line.strip())
    vectors = load_vectors()
    return trainList, vectors

def load_data_val_10(testList, vectors, index):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    items = testList[index].split(' ')
    x_train_1.append(vocab_plus_overlap(vectors, items[2], items[3], 200))
    x_train_2.append(vocab_plus_overlap(vectors, items[3], items[2], 200))
    x_train_3.append(vocab_plus_overlap(vectors, items[3], items[2], 200))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def read_raw():
    raw = []
    for line in open('datasets/insuranceQA/train'):
        items = line.strip().split(' ')
        query = items[2].split('_')
        if '<a>' in query:
            query = query[:query.index('<a>')]
        answer = items[3].split('_')
        if '<a>' in answer:
            answer = answer[:answer.index('<a>')]
        raw.append([query,answer,items[0]])
    neg = []
    for i,sample in enumerate(raw):
        for j in range(1):
            index = random.randint(0, len(raw) - 1)
            neg.append([sample[0],raw[index][1],'0'])
    raw.extend(neg)
    return raw

def encode_sent(vocab, string, size):
    x = []
    words = string.split('_')
    for i in range(0, size):
        if words[i] in vocab:
            x.append(vocab[words[i]])
        else:
            x.append(vocab['UNKNOWN'])
    return x

def load_data_6(vocab, alist, raw, size, seq_size=200):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    for i in range(0, size):
        items = raw[random.randint(0, len(raw) - 1)]
        nega = rand_qa(alist)
        x_train_1.append(encode_sent(vocab, items[2], seq_size))
        x_train_2.append(encode_sent(vocab, items[3], seq_size))
        x_train_3.append(encode_sent(vocab, nega, seq_size))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def load_data_val_6(testList, vocab, index, batch, seq_size=200):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    for i in range(0, batch):
        true_index = index + i
        if (true_index >= len(testList)):
            true_index = len(testList) - 1
        items = testList[true_index].split(' ')
        x_train_1.append(encode_sent(vocab, items[2], seq_size))
        x_train_2.append(encode_sent(vocab, items[3], seq_size))
        x_train_3.append(encode_sent(vocab, items[3], seq_size))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def load_data_9(trainList, vectors, size):
    x_train_1 = []
    x_train_2 = []
    y_train = []
    for i in range(0, size):
        pos = trainList[random.randint(0, len(trainList) - 1)]
        posItems = pos.strip().split(' ')
        x_train_1.append(vocab_plus_overlap(vectors, posItems[2], posItems[3], 200))
        x_train_2.append(vocab_plus_overlap(vectors, posItems[3], posItems[2], 200))
        y_train.append([1, 0])
        neg = trainList[random.randint(0, len(trainList) - 1)]
        negItems = neg.strip().split(' ')
        x_train_1.append(vocab_plus_overlap(vectors, posItems[2], negItems[3], 200))
        x_train_2.append(vocab_plus_overlap(vectors, negItems[3], posItems[2], 200))
        y_train.append([0, 1])
    return np.array(x_train_1), np.array(x_train_2), np.array(y_train)

def load_data_val_9(testList, vectors, index):
    x_train_1 = []
    x_train_2 = []
    items = testList[index].split(' ')
    x_train_1.append(vocab_plus_overlap(vectors, items[2], items[3], 200))
    x_train_2.append(vocab_plus_overlap(vectors, items[3], items[2], 200))
    return np.array(x_train_1), np.array(x_train_2)

def load_data_10(vectors, qalist, raw, size):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    items = raw[random.randint(0, len(raw) - 1)]
    nega = rand_qa(qalist)
    x_train_1.append(vocab_plus_overlap(vectors, items[2], items[3], 200))
    x_train_2.append(vocab_plus_overlap(vectors, items[3], items[2], 200))
    x_train_3.append(vocab_plus_overlap(vectors, nega, items[2], 200))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def load_data_11(vectors, qalist, raw, size):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    items = raw[random.randint(0, len(raw) - 1)]
    nega = rand_qa(qalist)
    x_train_1.append(vocab_plus_overlap(vectors, items[2], items[3], 200))
    x_train_2.append(vocab_plus_overlap(vectors, items[3], items[2], 200))
    x_train_3.append(vocab_plus_overlap(vectors, nega, items[2], 200))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def clean_train(data):
    result = []
    for k,v in data.items():
        for sample in v:
            result.append([k.split(' '),sample[0].split(' '),sample[1]])
    return result


def insurance_read_vocab():
    vocab = {}
    vocab2 = {}
    with open('datasets/insuranceQA_new/V2/vocabulary') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.replace('\n','').split('\t')
            vocab[line[0]]=line[1]
            vocab2[line[1]]=int(line[0][4:])
    return vocab,vocab2


def insurance_read_sen2id(vocab):
    sen2id = {}
    with open('datasets/insuranceQA_new/V2/InsuranceQA.label2answer.token.encoded') \
            as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.replace('\n','').split('\t')
            sen2id[line[0]]=[vocab[i] for i in line[1].split(' ')]
    return sen2id

def insurance_read_train(vocab,sen2id):
    train_set = []
    with open('datasets/insuranceQA_new/V2/InsuranceQA.quest'
              'ion.anslabel.token.100.pool.solr.train.encoded') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.replace('\n','').split('\t')
            query = [vocab[i] for i in line[1].split(' ')]
            for i in line[3].split(' '):
                train_set.append([query,sen2id[i],0])
            pos = []
            while len(pos)< len(line[3].split(' ')):
                for i in line[2].split(' '):
                    pos.append([query,sen2id[i],1])
            train_set.extend(pos)
    return train_set


def insurance_read_test(vocab,sen2id):
    train_set = []
    with open('datasets/insuranceQA_new/V2/InsuranceQA.quest'
              'ion.anslabel.token.100.pool.solr.test.encoded') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.replace('\n','').split('\t')
            query = [vocab[i] for i in line[1].split(' ')]
            for i in line[2].split(' '):
                train_set.append([query, sen2id[i], 1])
            for i in line[3].split(' '):
                train_set.append([query,sen2id[i],0])
    return train_set
