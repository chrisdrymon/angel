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

json_files = os.listdir('jsons')

# First check if the processed data already exists. This input data, however, has not been split into time-series
# samples yet.
if 'lstm_2_inputs-5-deep.pickle' in json_files and 'lstm_2_labels-5-deep.pickle' in json_files:
    lstm_2_pre_time_series = pickle.load(open(os.path.join('jsons', 'lstm_2_inputs_5-deep.pickle'), 'rb'))
    labels = pickle.load(open(os.path.join('jsons', 'lstm_2_labels-5-deep.pickle'), 'rb'))

# If the data does not exist, then we must create it.
else:
    corpora_folder = os.path.join('corpora', 'greek', 'annotated', 'perseus-771dca2', 'texts')
    indir = os.listdir(corpora_folder)
    file_count = 0
    train_data = []
    lstm1_inputs = []
    missing_label = 0
    dash_label = 0
    token_count = 0

    # Labels are of the form [sentence1->[pos, pos, ...], sentence2->[pos, pos, ...], ...]
    labels = []
    lstm_2_pre_time_series = []

    with open(os.path.join('jsons', 'all_characters.json'), encoding='utf-8') as json_file:
        all_characters = json.load(json_file)

    pos, person, number, tense, mood, voice, gender, case, degree = create_morph_classes()
    morphs = (pos, person, number, tense, mood, voice, gender, case, degree)

    # Change this for each aspect of morphology to be trained
    target_morphology = pos

    # Search through every work in the annotated Greek folder
    for file in indir[:5]:
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
                dnn_input_array = []
                lstm_2_sentence_array = []
                for token in tokens:

                    # Elliptical tokens were fine for training the character-reading NN's so long as they contained
                    # both a form and postags, but they're bad for training the next LSTM. Fortunately, within AGDT, the
                    # only tokens with missing wordforms or postags are artificial tokens. They can all be ignored
                    # without damaging time-series order because all but two of them occur at the end of the sentence.
                    # The two exceptions can safely be ignored as well. I manually checked.
                    if token.has_attr('artificial') is False:

                        # Create the labels. One morphology aspect per run, so only need one morph's labels.
                        label_tensor = [0] * (len(target_morphology.tags) + 1)
                        postag_index = morphs.index(target_morphology)
                        try:
                            label_tensor[target_morphology.tags.index(token['postag'][postag_index])] = 1
                        except IndexError:
                            print(sentence['id'], token['id'], token['form'])
                            label_tensor[-1] = 1
                            missing_label += 1
                        except ValueError:
                            print(sentence['id'], token['id'], token['form'])
                            label_tensor[-1] = 1
                            dash_label += 1
                        label_tensor = np.array(label_tensor)
                        labels.append(label_tensor)
                        token_count += 1

    # These labels will be used to train LSTM 2
    labels = np.array(labels, dtype=np.bool_)
    pickle.dump(labels, open(os.path.join('jsons', 'lstm_2_labels-5-deep.pickle'), 'wb'))
print(f'{missing_label} missing labels.')
print(f'{dash_label} "-" labels.')
print(f'{token_count} total tokens.')
