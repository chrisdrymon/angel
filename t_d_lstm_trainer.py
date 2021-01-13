import os
from bs4 import BeautifulSoup
import time
import json
import pickle
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from morph_utils import create_morph_classes, ModelSaver

# This setting keeps Tensorflow from automatically reserving all my GPU's memory
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

corpora_folder = os.path.join('corpora', 'greek', 'annotated', 'perseus-771dca2', 'texts')
indir = os.listdir(corpora_folder)
file_count = 0
train_data = []
lstm1_inputs = []

# Labels are of the form [sentence1->[correct_pos, correct_pos, ...], sentence2->[correct_pos, correct_pos, ...], ...]]
labels = []
dnn_input_array = []
td_lstm_input_array = []

with open(os.path.join('jsons', 'all_characters.json'), encoding='utf-8') as json_file:
    all_characters = json.load(json_file)

pos, person, number, tense, mood, voice, gender, case, degree = create_morph_classes()
morphs = (pos, person, number, tense, mood, voice, gender, case, degree)

# Change this for each aspect of morphology to be trained
target_morphology = pos
corpus_size = 1

# Search through every work in the annotated Greek folder
for file in indir[:corpus_size]:
    if file[-4:] == '.xml':
        file_count += 1
        print(file_count, file)

        # Open the files (they are XML's) with beautiful soup and search through every word in every sentence.
        xml_file = open(os.path.join(corpora_folder, file), 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'xml')
        sentences = soup.find_all('sentence')

        # The data is going to be processed and stored sentence by sentence. This will be slow, but will make the
        # creation of time-series input data much easier to keep track of.
        for sentence in sentences:
            tokens = sentence.find_all(['word', 'token'])
            sentence_of_inputs = []
            sentence_of_labels = []
            for token in tokens:

                # Elliptical tokens were fine for training the character-reading NN's so long as they contained both a
                # form and postags, but they're bad for the sentence-level LSTM training. Fortunately, within AGDT, the
                # only tokens with missing wordforms or postags are artificial tokens. They can all be ignored without
                # damaging time-series order because all but two of them occur at the end of the sentence. The two
                # exceptions can safely be ignored as well. I manually checked.
                if token.has_attr('artificial') is False:

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
                    sentence_of_inputs.append(token_tensor)

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
                    sentence_of_labels.append(label_tensor)
            sentence_of_inputs = np.array(sentence_of_inputs, dtype=np.bool_)
            lstm1_inputs.append(sentence_of_inputs)
            labels.append(sentence_of_labels)

# Convert the wordform samples to a numpy array
lstm1_inputs = np.array(lstm1_inputs, dtype=np.bool_)

print('Running LSTM samples through LSTMs. This will take a minute...')
# Run the wordform tensors through each of the LSTM's. Each of their softmax outputs will be used to train the target
# morph's DNN.
for morph in morphs:
    morph.predicted = morph.lstm.predict(lstm1_inputs)
    print(f'{morph.title} LSTM predictions complete...')

# Predicted softmax arrays are concatenated before input into the DNN.
print('Concatenating LSTM output tensors...')
i = 0
while i < len(pos.predicted):
    concatted_lstm_outputs = np.concatenate((pos.predicted[i], person.predicted[i], number.predicted[i],
                                             tense.predicted[i], mood.predicted[i], voice.predicted[i],
                                             gender.predicted[i], case.predicted[i], degree.predicted[i]))
    dnn_input_array.append(concatted_lstm_outputs)
    i += 1

dnn_input_array = np.array(dnn_input_array)

# Run inputs through the DNN.
print('Concatenations complete...')
print('Running outputs through DNNs...')
for morph in morphs:
    morph.dnn_output = morph.dnn.predict(dnn_input_array)
    print(f'{morph.title} DNN predictions complete...')

# Now take DNN softmax output from each morphology aspect and concatenate them together. This concatenated tensor is
# still representing one token. So now for each token, take timesteps from a [-10, 10] windows around the token and use
# that data to train a new LSTM for each aspect.
print('Concatenating DNN output tensors...')
i = 0
while i < len(pos.dnn_output):
    concatted_dnn_outputs = np.concatenate((pos.dnn_output[i], person.dnn_output[i], number.dnn_output[i],
                                            tense.dnn_output[i], mood.dnn_output[i], voice.dnn_output[i],
                                            gender.dnn_output[i], case.dnn_output[i], degree.dnn_output[i]))
    td_lstm_input_array.append(concatted_dnn_outputs)
    i += 1

print('Creating LSTM 2 training data...')
# Search through every work in the annotated Greek folder. We have to go back through all this because I want the
# tensors to be based on sentences.
td_array_index = 0

# This is the size of concatenated morphology tensors
blank_concat_tensor = np.array([0] * 55, dtype=np.bool_)
for file in indir[:corpus_size]:
    if file[-4:] == '.xml':
        file_count += 1
        print(file_count, file)

        # Open the files (they are XML's) with beautiful soup and search through every word in every sentence.
        xml_file = open(os.path.join(corpora_folder, file), 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'xml')
        sentences = soup.find_all('sentence')
        for sentence in sentences:
            toks = sentence.find_all(['word', 'token'])
            tokens = []

            # I have to deal with tokens which have missing forms or postags. There are 3 options: 1) Fill them with a
            # blank, 2) Skip them entirely. Act like they were never in the sentence, 3) Skip the whole sentence so as
            # to not give contaminated training data. The first option may confuse a trainer into thinking the next
            # word is at the beginning of a sentence. The second option will shift the word order of the rest of
            # the sentence. As this is a fairly rare occurrence, I think 3 is the best option. That's what I'm going to
            # do. I'll have to go back to the beginning of the program and change some code.
            # There's another problem: Elliptical tokens were fine for training the character-reading NN's so long as
            # they contained both a form and postags, but they're bad for the sentence-level LSTM training. They need
            # to be left out. I have to go back to the beginning and fix that.
            for tok in toks:
                if tok.has_attr('form') and tok.has_attr('postag'):
                    tokens.append(tok)
            for token in tokens:
                # This tensor will hold the entire time series input for this token
                one_time_series = []
                adder = -10


                    # Record blank tensors for samples that occur before the first token of a sentence
                    while i + adder < 0:
                        one_time_series.append(blank_concat_tensor)

                    while adder <= 10:
                        try:
                            one_time_series.append(td_lstm_input_array[td_array_index])
                        except KeyError:
                            one_time_series.append(blank_concat_tensor)

                    td_array_index += 1


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
