import json 
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import pandas as pd
import shutil
import re
import unicodedata
import itertools
import time
import random
import config
from evals import evaluate_generation

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(config.device)
    return loss, nTotal.item()

def evaluate(model, data):
    """
    evaluate
    """
    model.eval()
    ss = []
    with torch.no_grad():
        for data_batch in data:
            if data_batch[2].shape[1] == config.batch_size:
                avg_loss = model.iterate(data_batch=data_batch, is_training=False)
                ss.append(avg_loss)
    return sum(ss)/len(ss)

class Trainer(object):
    """docstring for trainer"""
    def __init__(self, model, optimizer, trainData, validData, vocab, num_epochs=1,
        generator=None,save_dir=None, log_steps=None,valid_steps=None, 
        grad_clip=None):
        self.model = model
        self.vocab = vocab
        self.optimizer = optimizer
        self.trainData = trainData
        self.validData = validData

        self.generator = generator
        self.save_dir = save_dir
        self.log_steps = log_steps
        self.valid_steps = valid_steps
        self.grad_clip = grad_clip
        self.num_epochs = num_epochs
        self.epoch = 0
        self.batch_num = 0

        self.train_start_message = "\n".join(["",
                                              "=" * 85,
                                              "=" * 34 + " Model Training " + "=" * 35,
                                              "=" * 85,
                                              ""])
        self.valid_start_message = "\n" + "-" * 33 + " Model Evaulation " + "-" * 33

    def train_epoch(self):
        self.epoch += 1
        num_batches = len(self.trainData)
        print(self.train_start_message)

        for batch_id, data_batch in enumerate(self.trainData,1):
            if data_batch[2].shape[1] == config.batch_size:
                self.model.train()
                start_time = time.time()
                avg_loss = self.model.iterate(data_batch,
                                            optimizer=self.optimizer,
                                            grad_clip=self.grad_clip,
                                            is_training=True)
                elapsed = time.time()-start_time
                self.batch_num += 1

                if batch_id % self.log_steps == 0:
                    message_prefix = "[Train][{:2d}][{}/{}]".format(self.epoch, batch_id, num_batches)
                    metrics_message = "Average Loss: "+str(avg_loss)
                    message_posfix = "TIME-{:.2f}".format(elapsed)
                    print("   ".join(
                        [message_prefix, metrics_message, message_posfix]))

                if batch_id % self.valid_steps == 0:
                    print(self.valid_start_message)
                    valid_loss = evaluate(self.model, self.validData)

                    message_prefix = "[Valid][{:2d}][{}/{}]".format(self.epoch, batch_id, num_batches)
                    metrics_message = "Valid Loss: "+str(valid_loss)
                    print("   ".join([message_prefix, metrics_message]))

                    if valid_loss < self.best_valid_loss:
                        self.best_valid_loss = valid_loss
                        self.save(True)
                    print("-" * 85 + "\n")

                    if self.generator is not None:
                        print("Generation starts ...")
                        gen_save_file = os.path.join(
                            self.save_dir, "valid_{}.result").format(self.epoch)
                        evaluate_generation(generator=self.generator,
                                                               data_iter=self.validData,
                                                               vocab=self.vocab,
                                                               save_file=gen_save_file)
                
        self.save()

    def train(self):
        self.best_valid_loss = evaluate(self.model, self.validData)
        print("Valid Loss: ", self.best_valid_loss)
        for _ in range(self.epoch, self.num_epochs):
            self.train_epoch()

    def save(self, is_best=False):
        model_file = os.path.join(
            self.save_dir, "state_epoch_{}.model".format(self.epoch))
        torch.save(self.model.state_dict(), model_file)
        print("Saved model state to '{}'".format(model_file))

        train_file = os.path.join(
            self.save_dir, "state_epoch_{}.train".format(self.epoch))
        train_state = {"epoch": self.epoch,
                       "batch_num": self.batch_num,
                       "best_valid_loss": self.best_valid_loss,
                       "optimizer": self.optimizer.state_dict()}
        torch.save(train_state, train_file)
        print("Saved train state to '{}'".format(train_file))

        if is_best:
            best_model_file = os.path.join(self.save_dir, "best.model")
            best_train_file = os.path.join(self.save_dir, "best.train")
            shutil.copy(model_file, best_model_file)
            shutil.copy(train_file, best_train_file)
            print(
                "Saved best model state to '{}' with new best valid loss {:.3f}".format(
                    best_model_file, self.best_valid_loss))

    def load(self, file_prefix):
        """
        load
        """
        model_file = "{}.model".format(file_prefix)
        train_file = "{}.train".format(file_prefix)

        model_state_dict = torch.load(
            model_file, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(model_state_dict)
        print("Loaded model state from '{}'".format(model_file))

        train_state_dict = torch.load(
            train_file, map_location=lambda storage, loc: storage)
        self.epoch = train_state_dict["epoch"]
        self.best_valid_loss = train_state_dict["best_valid_loss"]
        self.batch_num = train_state_dict["batch_num"]
        self.optimizer.load_state_dict(train_state_dict["optimizer"])

        print(
            "Loaded train state from '{}' with (epoch-{} best_valid_loss-{:.3f})".format(
                train_file, self.epoch, self.best_valid_loss))
    