import torch

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unkonw token

use_cuda = torch.cuda.is_available()
# use_cuda = False
device = torch.device("cuda:0" if use_cuda else "cpu")


conv_path = '../data'
wiki_path = '../WikiData'
save_dir = '../data/pointerKnow'
# ckpt = '../data/pointerKnow/model/state_epoch_100'
ckpt = None

# Configure models
model_name = 'cb_model'
# attn_model = 'dot'
#attn_model = 'general'
attn_model = 'concat'
pointer_gen = True
MAX_HISTORY_LENGTH = 300 #Maximum history length 
MAX_SEC_LENGTH = 300
MAX_LENGTH = 30  # Maximum sentence length to consider
MIN_COUNT = 3    # Minimum word count threshold for trimming
hidden_size = 300
embedding_size = 100
encoder_num_layers = 2
decoder_num_layers = 1
dropout = 0.3
batch_size = 16

# Configure training/optimization
optimizer = 'Adam'
grad_clip = 10.0
teacher_forcing_ratio = 1
lr = 0.0001
num_epochs = 100
log_steps = 100
valid_steps = 400

#Configure generator
beam_size = 5