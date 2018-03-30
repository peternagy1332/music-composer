# Algorithmic music generation using recurrent neural networks

## Summary

The model is capable to learn the style of a given artist and continue a previously unseen music in that style.

Since it learns based on WAV files with high dynamic range, the output is quite noisy as well. The neural net receives a random clip of your training dataset and emits a possible continuation which will be fed into the network again, and so on. That's why the noise is becoming louder and louder. Nonetheless, if the music were written for e.g. piano, better results could be accomplished, since that would be more "simple" for the net.

If you run python3 music-composer.py without further arguments, it will create a folder for you (input_music_dataset) and halt. Afterwards, copy your *.wav (not *.WAV) files into this folder, and run the script again. Please keep in mind that this task requires an enormous amount of computing power, so run it on powerful GPUs for as long as possible (e.g. 8 GB+ is recommended - otherwise the default neural model might not fit into the memory of your GPU!). The aim is to overfit the model with real music parts and generate music samples by feeding in different seeds.

## Getting started

### Requirements

- Minimum required Python version: 3.5.2
- Some *.wav files to train in a folder of the project root with 44100 Hz sampling rate.

### Setup

```bash
$ python3 -m venv mc
$ source mc/bin/activate
$ git clone https://github.com/peternagy1332/music-composer.git
$ cd music-composer
$ pip install -r requirements.txt

```

### Let the machine do the art

```bash
$ python3 music-composer.py -h

usage: music-composer.py [-h] [-d DATASETS_FOLDER_PATH]
                         [-t TRAINDATA_FOLDER_PATH] [-o OUTPUT_MUSIC_NAME]
                         [-n HIDDEN_LAYER_NEURONS] [-l HIDDEN_LAYER_COUNT]
                         [-e EPOCHS] [-b BATCH_SIZE] [-s SECONDS_TO_GENERATE]
                         [-g GENERATED_MUSIC_NUM]

Algorithmic music generator using a recurrent neural network

optional arguments:
  -h, --help            show this help message and exit
  -d DATASETS_FOLDER_PATH, --datasets-folder-path DATASETS_FOLDER_PATH
                        A folder for your *.wav files.
                        Default: input_music_dataset/
  -t TRAINDATA_FOLDER_PATH, --traindata-folder-path TRAINDATA_FOLDER_PATH
                        An empty folder for caching the preprocessed *.wav
                        files.
                        Default: train_data/
  -o OUTPUT_MUSIC_NAME, --output-music-name OUTPUT_MUSIC_NAME
                        The name of the output music.
                        Default: output
  -n HIDDEN_LAYER_NEURONS, --hidden-layer-neurons HIDDEN_LAYER_NEURONS
                        The number of LSTM neurons in each hidden layer.
                        Default: 1024
  -l HIDDEN_LAYER_COUNT, --hidden-layer-count HIDDEN_LAYER_COUNT
                        The number of hidden layers.
                        Default: 4
  -e EPOCHS, --epochs EPOCHS
                        The number of total epochs.
                        Default: 20
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The size of a training batch.
                        Default: 10
  -s SECONDS_TO_GENERATE, --seconds-to-generate SECONDS_TO_GENERATE
                        The length of the generated music in seconds.
                        Default: 10
  -g GENERATED_MUSIC_NUM, --generated-music-num GENERATED_MUSIC_NUM
                        How many music samples to generate.
                        Default: 10

```

It's worth to keep in mind that if you add or remove wav files, you should regenerate the training data by deleting the content of the traindata folder.

## The architecture
I used two time-distributed dense layers as input and output and LSTMs between them.

<img src="https://github.com/peternagy1332/music-composer/blob/master/assets/arch.png?raw=true" width="50%"/>

## Acknowledgement
Thanks for Matt Vitelli and Aran Nayebi for their remarkable work (https://github.com/MattVitelli/GRUV) that was the kickstarter of this project.
 