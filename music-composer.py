import glob
import random

import scipy.io.wavfile as wav
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
import os


def folder2examples(dataset_folder_path, traindata_folder_path, block_size, blocks_in_clip):
    print('Converting folder to examples ...')
    clips_x = []
    clips_y = []

    # lists of blocks
    block_list_x, block_list_y = dataset2blocks(dataset_folder_path, block_size, blocks_in_clip)

    current_block = 0
    total_blocks = len(block_list_x)
    while current_block + blocks_in_clip < total_blocks:
        clips_x.append(block_list_x[current_block:current_block + blocks_in_clip])
        clips_y.append(block_list_y[current_block:current_block + blocks_in_clip])
        current_block += blocks_in_clip

    number_of_examples = len(clips_x)
    print('# of examples (clips): ' + str(number_of_examples))

    out_shape = (number_of_examples, blocks_in_clip, block_size)
    x_data = np.zeros(out_shape)
    y_data = np.zeros(out_shape)

    for n in range(number_of_examples):
        for i in range(blocks_in_clip):
            x_data[n][i] = clips_x[n][i]
            y_data[n][i] = clips_y[n][i]

    mean_x = np.mean(np.mean(x_data, axis=0), axis=0)  # Mean across num examples and num timesteps
    std_x = np.sqrt(np.mean(np.mean(np.abs(x_data - mean_x) ** 2, axis=0), axis=0))  # STD across num examples and num timesteps
    std_x = np.maximum(1.0e-8, std_x)  # Clamp variance if too tiny

    x_data[:][:] -= mean_x  # Mean 0
    x_data[:][:] /= std_x  # Variance 1
    y_data[:][:] -= mean_x  # Mean 0
    y_data[:][:] /= std_x  # Variance 1

    np.save(traindata_folder_path + 'examples_mean', mean_x)
    np.save(traindata_folder_path + 'examples_var', std_x)
    np.save(traindata_folder_path + 'examples_x', x_data)
    np.save(traindata_folder_path + 'examples_y', y_data)

    print('... converted!')


def dataset2blocks(dataset_folder_path, block_size, blocks_in_clip):
    print('Loading train examples ...')
    long_wav = np.array([])

    for wav_file in glob.glob(dataset_folder_path + '*.wav'):
        data, bitrate = wav2ndarray(wav_file)
        #data = data.reshape((1,data.shape[0]))
        print('Concatenating track with shape ' + str(data.shape) + ' to long wav. ', end='')
        long_wav = np.concatenate((long_wav, data), axis=0)
        print('New long wav shape: ' + str(long_wav.shape))

#    long_wav = long_wav.reshape((1, long_wav.shape[1]))
    print(long_wav)

    block_list_x = npaudio2blocks(long_wav, block_size)
    block_list_y = block_list_x[blocks_in_clip:]

    for i in range(blocks_in_clip):
        block_list_y.append(np.zeros(block_size))

    print('... train examples loaded!')

    return block_list_x, block_list_y


def npaudio2blocks(long_wav, block_size):
    block_lists = []
    total_samples = long_wav.shape[0]
    num_samples_so_far = 0

    while num_samples_so_far < total_samples:
        block = long_wav[num_samples_so_far:num_samples_so_far + block_size]
        if block.shape[0] < block_size:
            padding = np.zeros((block_size - block.shape[0]))
            block = np.concatenate((block, padding))

        block_lists.append(block)
        num_samples_so_far += block_size

    return block_lists


def spawn_network(blocks_in_clip, block_size, hidden_layer_neurons=1024, hidden_layer_count=1):
    print('Spawning network ...')

    model = Sequential()

    model.add(TimeDistributed(Dense(units=block_size), input_shape=(blocks_in_clip, block_size)))

    for hidden_layer in range(hidden_layer_count):
        model.add(LSTM(units=hidden_layer_neurons, input_shape=(blocks_in_clip, block_size), return_sequences=True))

    model.add(TimeDistributed(Dense(units=block_size), input_shape=(blocks_in_clip, block_size)))

    model.compile(loss='mean_squared_error', optimizer='adadelta')

    print('... network spawned!')

    return model


def get_seed_sequence(training_data):
    """A very simple seed generator. Copies a random example's first seed_length sequences as input to the generation algorithm"""
    print('Getting random training example to be continued ...')
    random_training_id = random.randint(0, training_data.shape[0] - 1)
    # td = training_data[random_training_data_id]
    # seed = td.reshape((1,td.shape[0],td.shape[1]))
    seed = training_data[random_training_id:random_training_id+1]
    print('... first seed OK!')
    return seed


def doArt(model, num_of_blocks_to_generate, training_data, variance, mean):
    """Extrapolates from a given seed sequence"""

    seed = get_seed_sequence(training_data=training_data)

    output = None

    for block_num in range(num_of_blocks_to_generate):
        print('Generating new block (' + str(block_num) + '/' + str(num_of_blocks_to_generate) + ')... ', end='')
        new_block = model.predict(seed)
        print(new_block.min(), new_block.max(), new_block)

        print('DONE!')
        if output is None:
            output = (new_block*variance) + mean
        else:
            output = np.concatenate((output, (new_block*variance)+mean), axis=1)
        seed = new_block

    print('Output generated!')

    output = output.reshape((1,output.shape[0]*output.shape[1]*output.shape[2]))

    return output


def wav2ndarray(filename):
    data = wav.read(filename)
    np_arr = data[1].astype(np.float) / 32767.0 # Normalize 16-bit input to [-1, 1] range
    return np_arr, data[0]


def ndarray2wav(X, sample_rate, filename):
    X = (X*32767.0).astype(np.int16) # Scale up to [-32768, 32767] 16-bit PCM
    wav.write(filename, sample_rate, X.T)


def save_extrapolation(filename, generated_sequence, sample_frequency, variance, mean):
    ndarray2wav(generated_sequence, sample_frequency, filename)

def main():
    dataset_folder_path = 'datasets/'
    traindata_folder_path = 'train_data/'
    generated_music_name = 'output'

    sampling_frequency = 44100
    clip_length = 1
    block_size = int(round(sampling_frequency / 4))
    print('Block size: ' + str(block_size))

    # sequence of blocks length
    blocks_in_clip = int(round((sampling_frequency * clip_length) / block_size))
    print('Blocks in clip: ' + str(blocks_in_clip))

    # if train data folder is empty, transform WAVs into examples
    if not os.listdir(traindata_folder_path):
        folder2examples(dataset_folder_path, traindata_folder_path, block_size, blocks_in_clip)

    model = spawn_network(block_size=block_size, blocks_in_clip=blocks_in_clip, hidden_layer_neurons=1024, hidden_layer_count=4)
    print(model.summary())

    print('Loading train data ...')
    x_train = np.load(traindata_folder_path + 'examples_x.npy')
    y_train = np.load(traindata_folder_path + 'examples_y.npy')
    x_mean = np.load(traindata_folder_path + 'examples_mean.npy')
    x_var = np.load(traindata_folder_path + 'examples_var.npy')
    print('... train data loaded!')

    print('# of examples: ' + str(x_train.shape[0]))

    if os.path.isfile('weights'):
        print('Loading existing weights ...')
        model.load_weights('weights')
        print('... weights loaded!')

    print('Training started...')
    history = model.fit(x_train, y_train, batch_size=10, epochs=20, verbose=1)
    print(history.history)
    model.save_weights('weights')
    print('... training stopped!')

    print('Art is happening ...')
    for i in range(10):
        output = doArt(model, num_of_blocks_to_generate=10, training_data=x_train, variance=x_var, mean=x_mean)
        save_extrapolation(generated_music_name + '-' + str(i) + '.wav', output, sample_frequency=sampling_frequency, variance=x_var, mean=x_mean)
    print('... art happened!')


if __name__ == '__main__':
    main()
