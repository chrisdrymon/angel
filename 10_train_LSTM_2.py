import os
from bs4 import BeautifulSoup
import time
import json
import pickle
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from utilities_morph import create_morph_classes, ModelSaver

# This approach won't bother with padding sentences. Instead, it will consider words on both sides of punctuation so
# long as they are within the set window. I would expect it to work better on whole documents rather than individual
# sentences.

# This setting keeps Tensorflow from automatically reserving all my GPU's memory
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

print('Creating morphology classes...')
pos, person, number, tense, mood, voice, gender, case, degree = create_morph_classes()
morphs = (pos, person, number, tense, mood, voice, gender, case, degree)

sample_string = 'AGDT-first26'
val_string = 'AGDT-last7'
aspect = pos

# Import samples
print('Loading samples and validation...')
with open(os.path.join('data', 'pickles', f'samples-LSTM2-{sample_string}.pickle'), 'rb') as infile1:
    train_data = pickle.load(infile1)
with open(os.path.join('data', 'pickles', f'samples-LSTM2-{val_string}.pickle'), 'rb') as infile1:
    val_data = pickle.load(infile1)

# Load different labels for each aspect of morphology
with open(os.path.join('data', 'pickles', f'labels-{aspect.title}-{sample_string}.pickle'), 'rb') as infile1:
    train_labels = pickle.load(infile1)
with open(os.path.join('data', 'pickles', f'labels-{aspect.title}-{val_string}.pickle'), 'rb') as infile2:
    val_labels = pickle.load(infile2)

print(f'Train data shape: {train_data.shape}')
print(f'Train labels shape: {train_labels.shape}')
print(f'Val data shape: {val_data.shape}')
print(f'Val labels shape: {val_labels.shape}')

print('Padding data...')
train_blank_array = np.array([0]*192)
label_blank_array = np.array([0]*13)
start_data_padding = np.tile(train_blank_array, (7, 1))
end_data_padding = np.tile(train_blank_array, (8, 1))
start_label_padding = np.tile(label_blank_array, (15, 1))

padded_train_data = np.concatenate((start_data_padding, train_data, end_data_padding), axis=0)
padded_train_labels = np.concatenate((start_label_padding, train_labels), axis=0)
padded_val_data = np.concatenate((start_data_padding, val_data, end_data_padding), axis=0)
padded_val_labels = np.concatenate((start_label_padding, val_labels), axis=0)

print('Constructing data generators...')
train_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(padded_train_data, padded_train_labels, 15,
                                                                sampling_rate=1)
val_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(padded_val_data, padded_val_labels, 15, sampling_rate=1)

# Check these settings
relevant_morph = pos
corpus_string = 'AGDTfirst26last7'
nn_type = 'lstm2'
nn_layers = 3
cells = 128

# Enter the samples and labels into Tensorflow to train a neural network
print('Creating neural network...')
model = tf.keras.Sequential()

# Create the model according to the number of layers chosen above.
layers_left = nn_layers
if layers_left > 1:
    model.add(layers.Bidirectional(layers.LSTM(cells, activation='tanh', dropout=0.3, return_sequences=True),
                                   input_shape=(15, 192)))
    layers_left -= 1
    while layers_left > 1:
        model.add(layers.Bidirectional(layers.LSTM(cells, activation='tanh', dropout=0.3, return_sequences=True)))
        layers_left -= 1
    model.add(layers.Bidirectional(layers.LSTM(cells, activation='tanh', dropout=0.3)))
else:
    model.add(layers.Bidirectional(layers.LSTM(cells, activation='tanh', dropout=0.3), input_shape=(15, 192)))
model.add(layers.Dense(len(relevant_morph.tags) + 1, activation='softmax'))

modelSaver = ModelSaver(relevant_morph.title, nn_type, nn_layers, cells, corpus_string)

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
model.fit(train_gen, epochs=40, validation_data=val_gen, verbose=2, callbacks=[modelSaver])