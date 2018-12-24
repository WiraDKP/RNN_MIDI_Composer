# RNN_MIDI_Composer
Training a LSTM on Indonesian Folk Songs in MIDI format to compose a new MIDI music.

# How to train model
Set your desired configuration in `train.py` then run
> python train.py

You can visualize the model performance using the `Music Composer.ipynb` notebook while training.
![](asset/Loss.gif)
Note: The program will keep running unless you interrupt it with `ctrl+C`.

# How to compose music
Use the `Music Composer.ipynb` notebook. Load the model, then set your desired configuration.
![](asset/Compose.gif)
In this example, mymusic.csv is generated. Use midicsv program to convert it back to midi file.

# References
http://www.fourmilab.ch/webtools/midicsv/
https://www.midieditor.org
http://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
https://midialdo.blogspot.com/2017/12/download-kumpulan-midi-lagu-lagu-daerah.html
