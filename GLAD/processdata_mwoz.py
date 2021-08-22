import os
import json
import logging
import zipfile
import requests
from tqdm import tqdm
from vocab import Vocab
from embeddings import GloveEmbedding, KazumaCharEmbedding
from dataset import Dataset, Ontology


root_dir = os.path.dirname(__file__)
data_dir = os.path.join(root_dir, 'data', 'woz')


# furl = 'https://mi.eng.cam.ac.uk/~nm480/woz_2.0.zip'
# fzip = os.path.join(data_dir, 'mwoz.zip')

# draw = os.path.join(data_dir, 'raw')
draw = '/home1/lsy2018/GLAD/data/woz/raw'
dann = '/home1/lsy2018/GLAD/data/woz/annoate'
# dann = os.path.join(data_dir, 'ann')

splits = ['dev', 'train', 'test']


# def download(url, to_file):
#     r = requests.get(url, stream=True)
#     with open(to_file, 'wb') as f:
#         for chunk in r.iter_content(chunk_size=1024):
#             if chunk:
#                 f.write(chunk)


def missing_files(d, files):
    return not all([os.path.isfile(os.path.join(d, '{}.json'.format(s))) for s in files])


if __name__ == '__main__':
    # if not os.path.isfile(fzip):
    #     if not os.path.isdir(data_dir):
    #         os.makedirs(data_dir)
    #     logging.warn('Download from {} to {}'.format(furl, fzip))
        # download(furl, fzip)

    # if missing_files(draw, splits):
    #     if not os.path.isdir(draw):
    #         os.makedirs(draw)
    #     with zipfile.ZipFile(fzip) as f:
    #         logging.warn('Extracting {} to {}'.format(fzip, draw))
    #         for split in splits:
    #             with f.open('woz_2.0/woz2_{}.json'.format(split)) as fin, open(os.path.join(draw, '{}.json'.format(split)), 'wb') as fout:
    #                 fout.write(fin.read())

    if missing_files(dann, files=splits + ['ontology', 'vocab', 'emb']):
        if not os.path.isdir(dann):
            os.makedirs(dann)
        dataset = {}
        ontology = Ontology()
        vocab = Vocab()
        vocab.word2index(['<sos>', '<eos>'], train=True)
        for s in splits:
            fname = '{}.json'.format(s)
            logging.warn('Annotating {}'.format(s))
            dataset[s] = Dataset.annotate_raw(os.path.join(draw, fname))
            dataset[s].numericalize_(vocab)
            ontology = ontology + dataset[s].extract_ontology()
            with open(os.path.join(dann, fname), 'wt') as f:
                json.dump(dataset[s].to_dict(), f)

                
        ontology.numericalize_(vocab)
        with open(os.path.join(dann, 'ontology.json'), 'wt') as f:
            json.dump(ontology.to_dict(), f)
        with open(os.path.join(dann, 'vocab.json'), 'wt') as f:
            json.dump(vocab.to_dict(), f)

        logging.warn('Computing word embeddings')
        embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
        E = []
        for w in tqdm(vocab._index2word):
            e = []
            for emb in embeddings:
                e += emb.emb(w, default='zero')
            E.append(e)
        with open(os.path.join(dann, 'emb.json'), 'wt') as f:
            json.dump(E, f)