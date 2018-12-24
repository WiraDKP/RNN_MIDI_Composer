import numpy as np
import pandas as pd
import pickle

import torch
from torch import nn
import torch.nn.functional as F

import time, math

# init_converter()
midi2note = pickle.load(open("conversion/midi2note.pkl", "rb"))
note2midi = pickle.load(open("conversion/note2midi.pkl", "rb"))

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('Training on GPU!')
else: 
    print('Training on CPU')

## RNN Architecture
class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))
      
    def forward(self, x, hidden):
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.n_hidden)    
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        
        if  train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden


## Training    
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {s:4.1f}s'


def train(net, data, model_name, batch_size=10, seq_length=50, lr=0.001, clip=5, print_every_n_step=50, save_every_n_step=5000):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    if train_on_gpu:
        net.cuda()
    
    n_chars = len(net.chars)
    counter = epoch = 0   
    loss_history = []
    start = time.time()    
    while True:
        epoch += 1
        h = net.init_hidden(batch_size)
        
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)            
            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            h = tuple([each.data for each in h])

            net.zero_grad()
            
            output, h = net(inputs, h)
            loss = criterion(output, targets.view(batch_size*seq_length))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            # loss stats
            if counter % print_every_n_step == 0:
                print(f"Epoch: {epoch:5} | Step: {counter:6} | Loss: {loss.item():.4f} | Elapsed Time: {time_since(start)}")

            if counter % save_every_n_step == 0:
                print(f"Epoch: {epoch:5} | Step: {counter:6} | Loss: {loss.item():.4f} | Elapsed Time: {time_since(start)}")
                print(" --- Save checkpoint ---")
                checkpoint = {
                    'n_hidden': net.n_hidden,
                    'n_layers': net.n_layers,
                    'state_dict': net.state_dict(),
                    'tokens': net.chars,
                    'loss_history': loss_history
                }
                torch.save(checkpoint, open(f"{model_name}/epoch_{epoch}.pth", 'wb'))                
        loss_history.append(loss.item())
            
            
def predict(net, char, h=None, top_k=None):
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)

    if train_on_gpu:
        inputs = inputs.cuda()

    h = tuple([each.data for each in h])
    out, h = net(inputs, h)

    p = F.softmax(out, dim=1).data
    if train_on_gpu:
        p = p.cpu()

    if top_k is None:
        top_ch = np.arange(len(net.chars))
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
    elif top_k == 1: 
        char = p.argmax().item()
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())        
    return net.int2char[char], h


def sample(net, size, prime='The', top_k=None):
        
    if train_on_gpu:
        net.cuda()
    else:
        net.cpu()
    
    net.eval()
    
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)
    chars.append(char)
    
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)
    

def one_hot_encode(arr, n_labels):
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot


def get_batches(arr, batch_size, seq_length):
    batch_size_total = batch_size * seq_length
    n_batches = len(arr)//batch_size_total
    
    arr = arr[:n_batches * batch_size_total]
    arr = arr.reshape((batch_size, -1))
    
    for n in range(0, arr.shape[1], seq_length):
        x = arr[:, n:n+seq_length]
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y    
    
## Preprocessing
def csv2seq(foldername, filename, note):
    with open(f"{foldername}/{filename}", "r") as f:
        line = [line.strip("\n").split(", ") for line in f if len(line.split(", "))==6]

    division = int(line[0][-1])
    scale = 1024/division # Normalize to 1024 division

    df = pd.DataFrame(line, columns=["track", "time", "tipe", "channel", "note", "velocity"])
    df = df.loc[df.tipe.isin(["Note_on_c", "Note_off_c"])]

    df.time = df.time.apply(lambda x: round(int(x)*scale))
    df.track = df.track.apply(int)
    df.note = df.note.apply(lambda x: midi2note[int(x)])
    df.velocity = df.velocity.apply(int)

    df.drop(["channel", "velocity"], axis=1, inplace=True)

    filename = filename.strip(".csv")
    note[filename] = {}
    for track in df.track.unique():
        df_on = df.loc[(df.tipe=="Note_on_c") & (df.track==track)]
        df_off = df.loc[(df.tipe=="Note_off_c") & (df.track==track)]
        df_on.durr = [df_off[(df_off.note==note) & (df_off.time > time)].iloc[0, 1] for time, note in zip(df_on.time.values, df_on.note.values)] - df_on.time
        df_on["next"] = df_on.time.diff().shift(-1).fillna(0)
        df_on.note = df_on.note + "-" + df_on.durr.apply(lambda x: str(int(x))) + "-" + df_on.next.apply(lambda x: str(int(x)))
        note[filename][track-1] = " ".join(df_on.note.values)    
    return note    


def seq2csv(seq, fname, channel):
    out = []
    out.append(f"0, 0, Header, 1, {len(seq.keys())}, 1024")
    
    for idx, track in enumerate(seq.keys()):
        out.append(f"{idx+1}, 0, Start_track")
        out.append(f'{idx+1}, 0, Title_t, "{idx+1}"')
        df = pd.DataFrame([item.split("-") for item in seq[track].split()], columns=["note", "durr", "next"])
        df.note = df.note.apply(lambda x: note2midi[x])
        df.durr = df.durr.apply(int)
        df.next = df.next.apply(int)
        df["time_on"] = df.next.cumsum().shift(1).fillna(0).apply(int)
        df["time_off"] = df.time_on + df.durr
        df["track"] = idx+1
        df["tipe_on"], df["tipe_off"] = "Note_on_c", "Note_off_c"
        df["channel"] = channel[idx]
        df["velocity_on"], df["velocity_off"] = 65, 0     
        
        out1 = df[["track", "time_on", "tipe_on", "channel", "note", "velocity_on"]]
        out2 = df[["track", "time_off", "tipe_off", "channel", "note", "velocity_off"]]
        out1.columns = ["track", "time", "tipe", "channel", "note", "velocity"]
        out2.columns = ["track", "time", "tipe", "channel", "note", "velocity"]
        df = out1.append(out2).sort_values("time")
        end_time = df.time.iloc[-1] + 1

        out.extend((df.track.apply(str) + ", " + df.time.apply(str) + ", " + df.tipe + ", " + df.channel.apply(str) + ", " + df.note.apply(str) + ", " + df.velocity.apply(str)).values)
        out.append(f"{idx+1}, {end_time}, End_track")
    out.append(f"0, 0, End_of_file")
    with open(f"{fname}.csv", "w") as f:
        for item in out:
            f.write(f"{item}\n")


def init_converter():
    df = pd.read_csv("conversion/conversion.csv")
    df.note = df.note.apply(lambda x: x.split("/")[0])
    midi2note = dict(zip(df.midi, df.note))
    midi2note[60] = "C4"
    midi2note[69] = "A4"
    note2midi = dict(zip(midi2note.values(), midi2note.keys()))

    pickle.dump(midi2note, open("conversion/midi2note.pkl", "wb"))
    pickle.dump(note2midi, open("conversion/note2midi.pkl", "wb"))            