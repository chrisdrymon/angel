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
lstm_2_input_array = []

with open(os.path.join('jsons', 'all_characters.json'), encoding='utf-8') as json_file:
    all_characters = json.load(json_file)

pos, person, number, tense, mood, voice, gender, case, degree = create_morph_classes()
morphs = (pos, person, number, tense, mood, voice, gender, case, degree)

# Change this for each aspect of morphology to be trained
target_morphology = pos
corpus_size = 'fullAGDT'

# Search through every work in the annotated Greek folder
for file in indir:
    if file[-4:] == '.xml':
        file_count += 1
        print(file_count, file)

        # Open the files (they are XML's) with beautiful soup and search through every word in every sentence.
        xml_file = open(os.path.join(corpora_folder, file), 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'xml')
        sentences = soup.find_all('sentence')

        # The data is going to be processed and stored sentence by sentence. This will be slow, but will make the
        # creation of time-series input data much easier to keep track of.
        for j, sentence in enumerate(sentences):
            tokens = sentence.find_all(['word', 'token'])
            sentence_of_inputs = []
            sentence_of_labels = []
            dnn_input_array = []
            lstm_2_sentence_array = []
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

            # These labels will be used later to train LSTM 2
            labels.append(sentence_of_labels)

            # Convert to a numpy array
            sentence_of_inputs = np.array(sentence_of_inputs, dtype=np.bool_)

            # Run each sentence through the LSTM and DNN.
            for morph in morphs:
                morph.lstm_output = morph.lstm.predict(sentence_of_inputs)

            # Predicted softmax arrays are concatenated before input into the DNN.
            i = 0
            while i < len(pos.lstm_output):
                concatted_lstm_outputs = np.concatenate((pos.lstm_output[i], person.lstm_output[i],
                                                         number.lstm_output[i], tense.lstm_output[i],
                                                         mood.lstm_output[i], voice.lstm_output[i],
                                                         gender.lstm_output[i], case.lstm_output[i],
                                                         degree.lstm_output[i]))
                dnn_input_array.append(concatted_lstm_outputs)
                i += 1

            dnn_input_array = np.array(dnn_input_array)

            # Run inputs through the DNN.
            for morph in morphs:
                morph.dnn_output = morph.dnn.predict(dnn_input_array)

            # Now take DNN softmax output from each morphology aspect and concatenate them together. This concatenated
            # tensor is still representing one token. So now for each token, take timesteps from a [-10, 10] windows
            # around the token and use that data to train a new LSTM for each aspect.
            i = 0
            while i < len(pos.dnn_output):
                concatted_dnn_outputs = np.concatenate((pos.dnn_output[i], person.dnn_output[i], number.dnn_output[i],
                                                        tense.dnn_output[i], mood.dnn_output[i], voice.dnn_output[i],
                                                        gender.dnn_output[i], case.dnn_output[i], degree.dnn_output[i]))
                lstm_2_sentence_array.append(concatted_dnn_outputs)
                i += 1

            lstm_2_input_array.append(lstm_2_sentence_array)

            # Give a little progress report
            if j % 100 == 0:
                print(f'Sentence {j} of {len(sentences)} complete.')

lstm_2_input_array = np.array(lstm_2_input_array)
labels = np.array(labels, dtype=np.bool_)

pickle.dump(lstm_2_input_array, open(os.path.join('jsons', f'lstm_2_inputs-{corpus_size}-deep.pickle'), 'wb'))
pickle.dump(lstm_2_input_array, open(os.path.join('jsons', f'lstm_2_labels-{corpus_size}-deep.pickle'), 'wb'))

# # Search through every work in the annotated Greek folder. We have to go back through all this because I want the
# # tensors to be based on sentences.
# td_array_index = 0
#
#
# samples = np.array(train_data)
# labels = np.array(labels, dtype=np.bool_)
#
# print('\nTime to train the DNN...')
# print(f'Samples: {len(samples)}')
# print(f'Labels: {len(labels)}')
#
# # Split data into an 80%/20% training/validation split.
# split = int(.8*len(labels))
# train_data = np.array(samples[:split])
# val_data = np.array(samples[split:])
# train_labels = np.array(labels[:split], dtype=np.bool_)
# val_labels = np.array(labels[split:], dtype=np.bool_)
#
# # Enter the samples and labels into Tensorflow to train a neural network
# model = tf.keras.Sequential()
# model.add(layers.Dense(20, input_dim=55, activation='relu'))
# model.add(layers.Dense(len(target_morphology.tags) + 1, activation='softmax'))
# modelSaver = ModelSaver(target_morphology.title, 'dnn')
#
# model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
# model.fit(train_data, train_labels, epochs=20, validation_data=(val_data, val_labels), verbose=2,
#           callbacks=[modelSaver])
