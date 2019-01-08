import numpy as np
from utils import csv2seq, CharRNN, train

from tqdm import tqdm
import os, plac


@plac.annotations(
    # Optional
    n_hidden=("Number of hidden unit", "option", "nh", int),
    n_layers=("Number of hidden layer", "option", "nl", int),
    bs=("Batch size", "option", "bs", int),
    seq_len=("Length of input sequence", "option", "sl", int),
    lr=("Learning rate", "option", "lr", float),
    d_out=("Dropout rate", "option", "do", float),
    save_every=("Number of steps for a model to be saved", "option", "se", int),
    print_every=("Number of steps that the training information (loss, etc.) will be printed", "option", "pe", int),
    name=("Folder Name for the model. It will create a new folder with this name if the folder is not found", "option", "o", str),
    midi_source_folder=("Folder Name for the data. It must have the .csv files in Midicsv format", "option", "i", str),
)
def main(n_hidden=64, n_layers=2, bs=128, seq_len=25, lr=0.001, d_out=0.2, save_every=20000, print_every=500,
         name='model', midi_source_folder = "dataset"):
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


if __name__ == '__main__':
    plac.call(main)
