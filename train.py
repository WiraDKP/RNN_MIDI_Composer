import torch
import numpy as np
from utils import csv2seq, CharRNN, train

from tqdm import tqdm
import os

# Config
n_hidden = 64   # Number of hidden unit
n_layers = 2    # Number of hidden layer
bs = 128        # batch size
seq_len = 25    # Length of sequence
lr = 0.001      # Learning Rate
d_out = 0.2     # Dropout rate
save_every = 20000  # Number of step to save a model
print_every = 500   # Number of step to print loss
name = 'model_short_indofolk'               # Model Folder Name
midi_source_folder = "music_in_csv_indo"    # Dataset Folder Name

# Initialize Folder
if not os.path.exists(name):
    os.makedirs(name)

# Initialize Data
note = {}
for fname in tqdm(os.listdir(midi_source_folder)):
    if fname not in [".DS_Store", '.ipynb_checkpoints']:
        note = csv2seq(midi_source_folder, fname, note)
text = " ".join([track for music in note.values() for track in music.values()])

chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
encoded = np.array([char2int[ch] for ch in text])

# Training
net = CharRNN(chars, n_hidden, n_layers, drop_prob=d_out)
print(net)
train(net, encoded, name, batch_size=bs, seq_length=seq_len, lr=lr, print_every_n_step=print_every, save_every_n_step=save_every)