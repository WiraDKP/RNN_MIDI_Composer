# RNN_MIDI_Composer
Training a LSTM on Indonesian Folk Songs in MIDI format to compose a new MIDI music. Have a listen:
- 2106 epochs: https://github.com/WiraDKP/RNN_MIDI_Composer/blob/master/sample/mymusic_2106_epochs.mid
- 4211 epochs: https://github.com/WiraDKP/RNN_MIDI_Composer/blob/master/sample/mymusic_4211_epochs.mid
- 13158 epochs: https://github.com/WiraDKP/RNN_MIDI_Composer/blob/master/sample/mymusic_13158_epochs.mid

## How to prepare your data
Convert your midi file into .csv using _Midicsv_<sup>[1]</sup>, and put them in a folder, by default, in the `dataset` folder. 

## How to train model
Set your desired configuration in `train.py` then run
> python train.py

You can visualize the model performance using the `Music Composer.ipynb` notebook while training.

Note: The program will keep running unless you interrupt it with `ctrl+C`.

### Parameters in Training configuration
- `n_hidden`
Number of hidden unit.
- `n_layers`
Number of hidden layer.
- `bs`
batch size
- `seq_len`
Length of input sequence.
- `lr`
Learning Rate
- `d_out`
Dropout rate
- `save_every`
Number of steps for a model to be saved
- `print_every`
Number of steps that the training information (loss, etc.) will be printed
- `name`
Folder Name for the model. It will create a new folder with this name if the folder is not found.
- `midi_source_folder`
Folder Name for the data. It must have the .csv files in _Midicsv_<sup>[1]</sup> format.

## How to compose music
Use the `Music Composer.ipynb` notebook. Load the model, then set your desired configuration.

In this example, `mymusic.csv` is generated. Use _Midicsv_<sup>[1]</sup> to convert it back to midi file, and you can open it with common midi player pr you can try _MidiEditor_<sup>[2]</sup>

### Parameters in Composing configuration
- `fname`
The name used for the generated music (.csv). You need to convert it back to .mid using _Midicsv_<sup>[1]</sup>
- `prime`
Prime for the RNN to compose the characters
- `top_k`
Take top k most probable prediction to randomly choose from. `top_k = 1` means that we always use the most probable character.<br>To be more creative, use `top_k` around 3-5. If top_k value is too large, the prediction may not follow the desired format to be converted back to .mid format.
- `compose_len`
Length of character to compose. One music note will need 8-14 characters. 
- `channel`
The midi channels and track number. For example, `[0, 1, 2]` means three channels, with each Track 0, 1 and 2.

## Troubleshooting
- If `Retry music composing...` keeps on popping

It is caused by our model does not follow the format. For example, we would want C5-512-1024, but the model generated C5--512-1024. You can try to use less channel, decrease `top_k`, decrease `compose_len`, train longer, or get more data. Less `top_k` helps because it will follow the proper format of the data instead of randomly generate characters. The same with longer training, and more data so that it can properly learn the format. Lower `compose_len`, instead, just to avoid this problem before it happens. Less channel is a must, the more you try to generate, the more chances that the model broke the format.
- If the model replicates the music from dataset

It is overfitting. You can try to decrease the model model complexity (less `n_hidden`, `n_layers`, `seq_len`), choose a model with lower epoch (higher loss model), or increase the `d_out`.
- If the generated music sounds gibberish

Your data may be too complex. Try a more homogenous data.

## Sample Result
I have done some preprocessing on a list of Indonesia folk songs in midi format<sup>[3]</sup> and it results in all those .csv files in the `dataset` folder. The trained model is in the `model` folder, and the music it generates is in the `sample` folder.

Here is the Loss history
![](asset/Loss.png)

Note: You does not have to push the Loss to minimum to generate a good music.

## References
This project will not succeed without these references. Thank you indeed!
- http://www.fourmilab.ch/webtools/midicsv/
- https://www.midieditor.org
- https://midialdo.blogspot.com/2017/12/download-kumpulan-midi-lagu-lagu-daerah.html
- https://github.com/udacity/deep-learning-v2-pytorch/tree/master/recurrent-neural-networks/char-rnn
