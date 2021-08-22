import os
import requests
import logging
import json
from tqdm import tqdm

from dataset import Dataset, Ontology
from embeddings import GloveEmbedding, KazumaCharEmbedding
from vocab import Vocab

root_dir = os.path.dirname(__file__)
print('root_dir',root_dir)         #空
data_dir = os.path.join(root_dir, 'data', 'woz')

data_raw = os.path.join(data_dir, 'raw')            #原始数据
data_annoate = os.path.join(data_dir, 'annoate')    #标注数据

splits = ['dev', 'train', 'test']
def missing_files(d, files):
    return not all([os.path.isfile(os.path.join(d, '{}.json'.format(s))) for s in files])

def download(url, to_file):
    r = requests.get(url, stream=True)
    with open(to_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
if __name__ == '__main__':
    #没有原始数据就创建文件夹下载数据
    if missing_files(data_raw, splits):
        if not os.path.isdir(data_raw):
            os.makedirs(data_raw)
        download('https://github.com/nmrksic/neural-belief-tracker/raw/master/data/woz/woz_train_en.json', os.path.join(data_raw, 'train.json'))
        download('https://github.com/nmrksic/neural-belief-tracker/raw/master/data/woz/woz_validate_en.json', os.path.join(data_raw, 'dev.json'))
        download('https://github.com/nmrksic/neural-belief-tracker/raw/master/data/woz/woz_test_en.json', os.path.join(data_raw, 'test.json'))

    #没有标注好的数据
    if missing_files(data_annoate, files=splits + ['ontology', 'vocab', 'emb']):
        if not os.path.isdir(data_annoate):
            os.makedirs(data_annoate)
        
        dataset = {}
        ontology = Ontology()
        vocab = Vocab()
        vocab.word2index(['<sos>', '<eos>'], train=True)
    for s in splits:
        file_name = '{}.json'.format(s)
        logging.warn('Annotating {}'.format(s))
        dataset[s] = Dataset.annotate_raw(os.path.join(data_raw, file_name))
        dataset[s].numericalize_(vocab)
        ontology = ontology + dataset[s].extract_ontology()
        with open(os.path.join(data_annoate, file_name), 'wt') as f:
            json.dump(dataset[s].to_dict(), f)
    ontology.numericalize_(vocab)
    with open(os.path.join(data_annoate, 'ontology.json'), 'wt') as f:
            json.dump(ontology.to_dict(), f)
    with open(os.path.join(data_annoate, 'vocab.json'), 'wt') as f:
        json.dump(vocab.to_dict(), f)

    logging.warn('Computing word embeddings')
    embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
    E = []
    for w in tqdm(vocab._index2word):
        e = []
        for emb in embeddings:
            e += emb.emb(w, default='zero')
        E.append(e)
    with open(os.path.join(data_annoate, 'emb.json'), 'wt') as f:
        json.dump(E, f)



