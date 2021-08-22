import os
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch

model_path = "/home1/lsy2018/NL2SQL/python5/chinese_L-12_H-768_A-12/"

if os.path.exists(model_path + "pytorch_model.bin") is False:
    convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
        model_path + 'bert_model.ckpt',
        model_path + 'bert_config.json',
        model_path + 'pytorch_model.bin')