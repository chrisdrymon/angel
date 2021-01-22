import os
from bs4 import BeautifulSoup
import time
import json
import pickle
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from utilities_morph import create_morph_classes, ModelSaver

# This setting keeps Tensorflow from automatically reserving all my GPU's memory
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

corpora_folder = os.path.join('corpora', 'greek', 'annotated', 'perseus-771dca2', 'texts')
indir = os.listdir(corpora_folder)
file_count = 0
train_data = []
wordform_tensors = []
labels = []

with open(os.path.join('jsons', 'all_characters.json'), encoding='utf-8') as json_file:
    all_characters = json.load(json_file)

pos, person, number, tense, mood, voice, gender, case, degree = create_morph_classes()
morphs = (pos, person, number, tense, mood, voice, gender, case, degree)

# Change this for each aspect of morphology to be trained
target_morphology = degree

# Search through every work in the annotated Greek folder
for file in indir[:5]:
    if file[-4:] == '.xml':
        file_count += 1
        print(file_count, file)

        # Open the files (they are XML's) with beautiful soup and search through every word in every sentence.
        xml_file = open(os.path.join(corpora_folder, file), 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'xml')
        sentences = soup.find_all('sentence')
        for sentence in sentences:
            tokens = sentence.find_all(['word', 'token'])
            for token in tokens:
                if token.has_attr('form') and token.has_attr('postag'):

                    # In the AGDT corpus, 218 unique characters occur. Hence the size of the character tensor.
                    blank_character_tensor = np.array([0]*219, dtype=np.bool_)

                    # The longest word in the AGDT corpus is 21 characters long
                    token_tensor = np.array([blank_character_tensor]*21, dtype=np.bool_)
                    wordform_length = len(token['form'])

                    # For each character in the token, create a one-hot tensor
                    for i, character in enumerate(token['form']):
                        character_tensor = np.array([0]*219, dtype=np.bool_)
                        try:
                            character_tensor[all_characters.index(character)] = 1
                        except ValueError:
                            character_tensor[218] = 1
                        token_tensor[21 - wordform_length + i] = character_tensor

                    # This tensor collects all the wordform tensors
                    wordform_tensors.append(token_tensor)

                    # We're only training one morphology aspect per run, so only need one morph's labels.
                    label_tensor = [0] * (len(target_morphology.tags) + 1)
                    postag_index = morphs.index(target_morphology)
                    try:
                        label_tensor[target_morphology.tags.index(token['postag'][postag_index])] = 1
                    except IndexError:
                        print(sentence['id'], token['id'], token['form'])
                        label_tensor[-1] = 1
                    except ValueError:
                        print(sentence['id'], token['id'], token['form'])
                        label_tensor[-1] = 1
                    labels.append(label_tensor)

# Convert the wordform samples to a numpy array
wordform_tensors = np.array(wordform_tensors)

print('Running LSTM samples through LSTMs. This will take a minute...')
# Run the wordform tensors through each of the LSTM's. Each of their softmax outputs will be used to train the target
# morph's DNN.
for morph in morphs:
    morph.lstm_output = morph.lstm.predict(wordform_tensors)
    print(f'{morph.title} LSTM predictions complete...')

# Predicted softmax arrays are concatenated before input into the DNN.
print('Concatenating LSTM output tensors...')
i = 0
while i < len(pos.lstm_output):
    concatted_lstm_outputs = np.concatenate((pos.lstm_output[i], person.lstm_output[i], number.lstm_output[i],
                                             tense.lstm_output[i], mood.lstm_output[i], voice.lstm_output[i],
                                             gender.lstm_output[i], case.lstm_output[i], degree.lstm_output[i]))
    train_data.append(concatted_lstm_outputs)
    i += 1
samples = np.array(train_data)
labels = np.array(labels, dtype=np.bool_)

print('\nTime to train the DNN...')
print(f'Samples: {len(samples)}')
print(f'Labels: {len(labels)}')

# Split data into an 80%/20% training/validation split.
split = int(.8*len(labels))
train_data = np.array(samples[:split])
val_data = np.array(samples[split:])
train_labels = np.array(labels[:split], dtype=np.bool_)
val_labels = np.array(labels[split:], dtype=np.bool_)

# Enter the samples and labels into Tensorflow to train a neural network
model = tf.keras.Sequential()
model.add(layers.Dense(20, input_dim=55, activation='relu'))
model.add(layers.Dense(len(target_morphology.tags) + 1, activation='softmax'))
modelSaver = ModelSaver(target_morphology.title, 'dnn')

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=20, validation_data=(val_data, val_labels), verbose=2,
          callbacks=[modelSaver])
