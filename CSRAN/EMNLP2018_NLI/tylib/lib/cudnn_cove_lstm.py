"""Defines an LSTM that loads CoVe weights using cuDNN.
"""

import functools
import numpy as np
import operator
import os
import tensorflow as tf
import cudnn_rnn_ops as old_cudnn

""" Constructs a Cudnn Cove LSTM, Loading
weights from CovE model.
"""

class CudnnCoveLstm:
    def __init__(self, weights_1, biases_1, weights_2, biases_2):
        self.num_units = 300
        self.input_dim = 300
        # Why not just one CudnnLSTM with 2 layers? I couldn't get that to
        # work correctly.
        self.lstm_1 = old_cudnn.CudnnLSTM(1,
            self.num_units, self.input_dim, direction="bidirectional",
            dropout=0.0)
        self.lstm_2 = old_cudnn.CudnnLSTM(1,
            self.num_units, self.num_units * 2, direction="bidirectional",
            dropout=0.0)

        self.buf_1 = self.lstm_1.canonical_to_params(weights_1, biases_1)
        self.buf_2 = self.lstm_2.canonical_to_params(weights_2, biases_2)

    def __call__(self, input_data):
        """Runs the CoVe cuDNN LSTM.

           Inputs:
            input_data: A tensor of shape [batch_size, max_len, 300]

           Output:
            A tensor of size [batch_size, max_len, 600].
        """
        batch_size = tf.shape(input_data)[0]
        input_h = tf.zeros([2, batch_size, self.num_units])
        input_c = tf.zeros([2, batch_size, self.num_units])
        output, output_h, output_c = \
            self.lstm_1(tf.transpose(input_data, [1, 0, 2]),
                input_h, input_c, self.buf_1, is_training=False)
        output, output_h, output_c = \
            self.lstm_2(output,
                input_h, input_c, self.buf_2, is_training=False)
        return tf.transpose(output, [1, 0, 2])

def _load_cove_np_arr(cove_file_name):
    np_value = np.load('./cove_weights/rnn.{}'.format(cove_file_name))
    return np.split(np_value, 4, axis=0)

def load_cudnn_cove_lstm():
    print("Attempting to load Cove Weights")
    weight_ih_l0 = _load_cove_np_arr("weight_ih_l0.npy")
    weight_hh_l0 = _load_cove_np_arr("weight_hh_l0.npy")
    bias_ih_l0 = _load_cove_np_arr("bias_ih_l0.npy")
    bias_hh_l0 = _load_cove_np_arr("bias_hh_l0.npy")
    weight_ih_l1 = _load_cove_np_arr("weight_ih_l1.npy")
    weight_hh_l1 = _load_cove_np_arr("weight_hh_l1.npy")
    bias_ih_l1 = _load_cove_np_arr("bias_ih_l1.npy")
    bias_hh_l1 = _load_cove_np_arr("bias_hh_l1.npy")
    weight_ih_l0_reverse = _load_cove_np_arr("weight_ih_l0_reverse.npy")
    weight_hh_l0_reverse = _load_cove_np_arr("weight_hh_l0_reverse.npy")
    bias_ih_l0_reverse = _load_cove_np_arr("bias_ih_l0_reverse.npy")
    bias_hh_l0_reverse = _load_cove_np_arr("bias_hh_l0_reverse.npy")
    weight_ih_l1_reverse = _load_cove_np_arr("weight_ih_l1_reverse.npy")
    weight_hh_l1_reverse = _load_cove_np_arr("weight_hh_l1_reverse.npy")
    bias_ih_l1_reverse = _load_cove_np_arr("bias_ih_l1_reverse.npy")
    bias_hh_l1_reverse = _load_cove_np_arr("bias_hh_l1_reverse.npy")
    print("Finished loading all Cove Weights..")
    weights_1 = \
        (
        weight_ih_l0 +
        weight_hh_l0 +
        weight_ih_l0_reverse +
        weight_hh_l0_reverse)
    weights_2 = \
        (weight_ih_l1 +
        weight_hh_l1 +
        weight_ih_l1_reverse +
        weight_hh_l1_reverse)
    biases_1 = \
        (bias_ih_l0 +
        bias_hh_l0 +
        bias_ih_l0_reverse +
        bias_hh_l0_reverse)
    biases_2 = \
        (bias_ih_l1 +
        bias_hh_l1 +
        bias_ih_l1_reverse +
        bias_hh_l1_reverse)

    return CudnnCoveLstm(weights_1, biases_1, weights_2, biases_2)
