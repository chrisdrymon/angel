import os
from bs4 import BeautifulSoup
import time
import json
import tensorflow as tf
import numpy as np


class Morphs:
    """Holds data for an aspect of morphology"""
    def __init__(self, title, tags, lstm):
        self.title = title
        self.tags = tags
        self.lstm = lstm


corpora_folder = os.path.join('corpora', 'greek', 'annotated', 'perseus-771dca2', 'texts')
models_folder = os.path.join('models')
indir = os.listdir(corpora_folder)
file_count = 0
accuracy_dict = {}

# Open the characters list
with open(os.path.join('jsons', 'all_characters.json'), encoding='utf-8') as json_file:
    all_characters = json.load(json_file)

# Load each trained model for testing
pos_LSTM = tf.keras.models.load_model(os.path.join('models', 'pos-1x64-0.945val0.907'))
pos_DNN = tf.keras.models.load_model(os.path.join('models', 'pos-DNN-1x20-0.925val0.914'))
person_LSTM = tf.keras.models.load_model(os.path.join('models', 'person-1x64-0.995val0.979'))
number_LSTM = tf.keras.models.load_model(os.path.join('models', 'number-1x64-0.990val0.967'))
tense_LSTM = tf.keras.models.load_model(os.path.join('models', 'tense-1x64-0.996val0.973'))
mood_LSTM = tf.keras.models.load_model(os.path.join('models', 'mood-1x64-0.995val0.978'))
voice_LSTM = tf.keras.models.load_model(os.path.join('models', 'voice-1x64-0.996val0.977'))
gender_LSTM = tf.keras.models.load_model(os.path.join('models', 'gender-1x64-0.962val0.909'))
case_LSTM = tf.keras.models.load_model(os.path.join('models', 'case-1x64-0.977val0.934'))
degree_LSTM = tf.keras.models.load_model(os.path.join('models', 'degree-1x64-0.999val0.999'))

# The possible tags for each item of morphology
pos_tags = ('l', 'n', 'a', 'r', 'c', 'i', 'p', 'v', 'd', 'm', 'g', 'u')
person_tags = ('1', '2', '3')
number_tags = ('s', 'p', 'd')
tense_tags = ('p', 'i', 'r', 'l', 't', 'f', 'a')
mood_tags = ('i', 's', 'n', 'm', 'p', 'o')
voice_tags = ('a', 'p', 'm', 'e')
gender_tags = ('m', 'f', 'n')
case_tags = ('n', 'g', 'd', 'a', 'v')
degree_tags = ('p', 'c', 's')

# Create a class instance for each aspect of morphology
pos = Morphs('pos', pos_tags, pos_LSTM)
person = Morphs('person', person_tags, person_LSTM)
number = Morphs('number', number_tags, number_LSTM)
tense = Morphs('tense', tense_tags, tense_LSTM)
mood = Morphs('mood', mood_tags, mood_LSTM)
voice = Morphs('voice', voice_tags, voice_LSTM)
gender = Morphs('gender', gender_tags, gender_LSTM)
case = Morphs('case', case_tags, case_LSTM)
degree = Morphs('degree', degree_tags, degree_LSTM)

# List each class in a tuple for iterating through later
morphs = (pos, person, number, tense, mood, voice, gender, case, degree)

total_tokens = 0

# Search through every work in the annotated Greek folder
for file in indir[5:]:
    if file[-4:] == '.xml':
        file_count += 1
        print(file_count, file)
        for morph in morphs:
            morph.correct_prediction_count = 0
            morph.labels = []
        accuracy_dict[file] = {}

        # A list to hold the token tensors
        samples = []
        token_count = 0

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
                    wordform_tensor = np.array([blank_character_tensor] * 21, dtype=np.bool_)
                    wordform_length = len(token['form'])

                    # For each character in the token, create a one-hot tensor
                    for i, character in enumerate(token['form']):
                        character_tensor = np.array([0]*219, dtype=np.bool_)
                        try:
                            character_tensor[all_characters.index(character)] = 1
                        except ValueError:
                            character_tensor[218] = 1
                        wordform_tensor[21 - wordform_length + i] = character_tensor

                    # This tensor collects all the wordform tensors
                    samples.append(wordform_tensor)

                    # Creates a labels tensor to check predictions against
                    for i, morph in enumerate(morphs):
                        try:
                            morph.labels.append(token['postag'][i])
                        except IndexError:
                            morph.labels.append('-')

                    # Keep a running total of qualifying tokens in each file
                    token_count += 1

        # Turn the samples into a numpy array
        samples = np.array(samples, dtype=np.bool_)

        # Get predictions from the model
        for morph in morphs:
            for i, predicted_tensor in enumerate(morph.lstm.predict(samples)):
                try:
                    predicted = morph.tags[int(np.argmax(predicted_tensor))]
                except IndexError:
                    predicted = '-'
                if predicted == morph.labels[i]:
                    morph.correct_prediction_count += 1
            morph.accuracy = morph.correct_prediction_count / token_count
            accuracy_dict[file][morph.title] = morph.accuracy
            print(f'{morph.title} accuracy: {morph.accuracy:.02%}')

        print(accuracy_dict)

# Save model accuracy to a file that can be compared with others later
with open(os.path.join('jsons', 'accuracy_records.json'), 'w') as outfile:
    json.dump(accuracy_dict, outfile)

# This is for assessing DNN's on top of the LSTMs
# i = 0
# while i < len(predicted_pos_tensors):
#     concatted = np.concatenate((predicted_pos_tensors[i], predicted_person_tensors[i],
#                                 predicted_number_tensors[i], predicted_tense_tensors[i],
#                                 predicted_mood_tensors[i], predicted_voice_tensors[i],
#                                 predicted_gender_tensors[i], predicted_case_tensors[i],
#                                 predicted_degree_tensors[i]))
#     i += 1
#     dnn_assessing.append(concatted)
# dnn_assess_array = np.array(dnn_assessing)
# predicted_pos_tensors = pos_DNN.predict(dnn_assess_array)
