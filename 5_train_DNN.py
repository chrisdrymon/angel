import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from utilities_morph import create_morph_classes, ModelSaver

# This setting keeps Tensorflow from automatically reserving all my GPU's memory
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

sample_corpus = 'AGDT-first26'
val_corpus = 'AGDT-last7'
nn_type = 'dnn'
nn_layers = 2
cells = 20
corpus_string = 'AGDTfirst26last7'

# The same set of samples and validation data can be loaded for the training of each aspect of morphology
print('Loading samples and validation data...')
with open(os.path.join('data', 'pickles', f'samples-DNN-{sample_corpus}.pickle'), 'rb') as infile:
    train_data = pickle.load(infile)
with open(os.path.join('data', 'pickles', f'samples-DNN-{val_corpus}.pickle'), 'rb') as infile:
    val_data = pickle.load(infile)

print('Creating morphology classes...')
pos, person, number, tense, mood, voice, gender, case, degree = create_morph_classes()
morphs = (pos, person, number, tense, mood, voice, gender, case, degree)

for aspect in morphs:

    # Appropriate label data needs to be loaded for each aspect of morphology
    print(f'Loading {aspect.title} labels...')
    with open(os.path.join('data', 'pickles', f'labels-{aspect.title}-{sample_corpus}.pickle'), 'rb') as infile:
        train_labels = pickle.load(infile)
    with open(os.path.join('data', 'pickles', f'labels-{aspect.title}-{val_corpus}.pickle'), 'rb') as infile:
        val_labels = pickle.load(infile)

    # Enter the samples and labels into Tensorflow to train a neural network
    model = tf.keras.Sequential()
    model.add(layers.Dropout(0.05, input_shape=(92,)))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dropout(0.05))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(len(aspect.tags) + 1, activation='softmax'))
    modelSaver = ModelSaver(aspect.title, nn_type, nn_layers, cells, corpus_string)

    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=20, validation_data=(val_data, val_labels), verbose=2,
              callbacks=[modelSaver])
