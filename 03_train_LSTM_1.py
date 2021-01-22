import os
import tensorflow as tf
from tensorflow.keras import layers
import pickle
import numpy as np
from utilities_morph import create_morph_classes, ModelSaver

# Enable this to run on CPU instead of GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# This setting keeps Tensorflow from automatically reserving all my GPU's memory
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

# Create the morphology classes
pos, person, number, tense, mood, voice, gender, case, degree = create_morph_classes()
morphs = (pos, person, number, tense, mood, voice, gender, case, degree)

# Load the datasets
print('Loading datasets...')
with open(os.path.join('data', 'pickles', 'samples-AGDT-first26.pickle'), 'rb') as outfile:
    train_data = pickle.load(outfile)
with open(os.path.join('data', 'pickles', 'labels-pos-AGDT-first26-tensors.pickle'), 'rb') as outfile:
    train_labels = pickle.load(outfile)
with open(os.path.join('data', 'pickles', 'samples-AGDT-last7.pickle'), 'rb') as outfile:
    val_data = pickle.load(outfile)
with open(os.path.join('data', 'pickles', 'labels-pos-AGDT-last7-tensors.pickle'), 'rb') as outfile:
    val_labels = pickle.load(outfile)

# train_labels = np.array(train_labels, dtype=np.bool_)
# val_labels = np.array(val_labels, dtype=np.bool_)

# Check these settings
relevant_morph = pos
corpus_string = 'AGDTfirst26last7'
nn_type = 'lstm1'
nn_layers = 2
cells = 128

# Enter the samples and labels into Tensorflow to train a neural network
print('Creating neural network...')
model = tf.keras.Sequential()

# Create the model layers.
layers_left = nn_layers
if layers_left > 1:
    model.add(layers.Bidirectional(layers.LSTM(cells, activation='tanh', dropout=0.3, return_sequences=True),
                                   input_shape=(21, 174)))
    layers_left -= 1
    while layers_left > 1:
        model.add(layers.Bidirectional(layers.LSTM(cells, activation='tanh', dropout=0.3, return_sequences=True)))
        layers_left -= 1
    model.add(layers.Bidirectional(layers.LSTM(cells, activation='tanh', dropout=0.3)))
else:
    model.add(layers.Bidirectional(layers.LSTM(cells, activation='tanh', dropout=0.3), input_shape=(21, 174)))
model.add(layers.Dense(len(relevant_morph.tags) + 1, activation='softmax'))

modelSaver = ModelSaver(relevant_morph.title, nn_type, nn_layers, cells, corpus_string)

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=40, validation_data=(val_data, val_labels), verbose=2,
          callbacks=[modelSaver])
