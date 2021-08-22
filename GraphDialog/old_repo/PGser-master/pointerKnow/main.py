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
from model import EncoderRNN,LuongAttnDecoderRNN,Copy_seq2seq
from decoder import GreedySearchDecoder,BeamSearchDecoder
from train import Trainer
from corpus import Corpus
import config
from evals import evaluate_generation


def main():
    dataCor = Corpus(config.conv_path,config.wiki_path,config.save_dir)
    dataCor.load()

    trainData = dataCor.create_batches(config.batch_size,'train')
    validData = dataCor.create_batches(config.batch_size,'valid')
    testData = dataCor.create_batches(config.batch_size,'test')


    model = Copy_seq2seq(vocab_size=dataCor.vocab.num_words,
                     embedding_size=config.embedding_size,
                     hidden_size=config.hidden_size,
                     encoder_num_layers=config.encoder_num_layers,
                     decoder_num_layers=config.decoder_num_layers,
                     dropout=config.dropout)
    if config.use_cuda:
        model.cuda(config.device)


    # Optimizer definition
    optimizer = getattr(torch.optim, config.optimizer)(
        model.parameters(), lr=config.lr)


    # Run training iterations
    print("Starting Training!")
    save_dir = os.path.join(config.save_dir,'model')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    generator = GreedySearchDecoder(model)

    print("Training starts ...")
    trainer = Trainer(model=model, optimizer=optimizer, trainData=trainData,
                      validData=validData,vocab=dataCor.vocab,generator=generator,num_epochs=config.num_epochs,
                      save_dir=save_dir, log_steps=config.log_steps,
                      valid_steps=config.valid_steps, grad_clip=config.grad_clip)
    if config.ckpt is not None:
        trainer.load(file_prefix=config.ckpt)
    trainer.train()

    result_file = os.path.join(config.save_dir,'greedy.txt')
    evaluate_generation(generator=generator,
                       data_iter=testData,
                       vocab=dataCor.vocab,
                       save_file=result_file)

    result_file = os.path.join(config.save_dir,'beamSearch.txt')
    generator = BeamSearchDecoder(model)
    evaluate_generation(generator=generator,
                       data_iter=testData,
                       vocab=dataCor.vocab,
                       save_file=result_file)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")