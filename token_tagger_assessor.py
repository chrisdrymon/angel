import os
from bs4 import BeautifulSoup
import time
import json
import datetime
import tensorflow as tf
import numpy as np


class Testing:
    """Provides data to assess model accuracy"""
    def __init__(self, tags, model):
        self.tags = tags
        self.model = model
        self.labels = []
        self.correct_prediction_count = 0


corpora_folder = os.path.join('corpora', 'greek', 'annotated', 'perseus-771dca2', 'texts')
models_folder = os.path.join('models')
indir = os.listdir(corpora_folder)
file_count = 0
accuracy_dict = {}

pos_model = tf.keras.models.load_model(os.path.join('models', 'pos-m1x64-0.944val0.906'))
pos_DNN = tf.keras.models.load_model(os.path.join('models', 'pos-DNN-m1x20-0.925val0.914'))
person_model = tf.keras.models.load_model(os.path.join('models', 'person-m1x64-0.995val0.979'))
number_model = tf.keras.models.load_model(os.path.join('models', 'number-m1x64-0.990val0.967'))
tense_model = tf.keras.models.load_model(os.path.join('models', 'tense-m1x64-0.996val0.973'))
mood_model = tf.keras.models.load_model(os.path.join('models', 'mood-m1x64-0.995val0.978'))
voice_model = tf.keras.models.load_model(os.path.join('models', 'voice-m1x64-0.996val0.977'))
gender_model = tf.keras.models.load_model(os.path.join('models', 'gender-m1x64-0.962val0.909'))
case_model = tf.keras.models.load_model(os.path.join('models', 'case-m1x64-0.977val0.934'))
degree_model = tf.keras.models.load_model(os.path.join('models', 'degree-m1x64-0.999val0.999'))

with open(os.path.join('jsons', 'all_characters.json'), encoding='utf-8') as json_file:
    all_characters = json.load(json_file)
pos_tags = ['l', 'n', 'a', 'r', 'c', 'i', 'p', 'v', 'd', 'm', 'g', 'u']
person_tags = ['1', '2', '3']
number_tags = ['s', 'p', 'd']
tense_tags = ['p', 'i', 'r', 'l', 't', 'f', 'a']
mood_tags = ['i', 's', 'n', 'm', 'p', 'o']
voice_tags = ['a', 'p', 'm', 'e']
gender_tags = ['m', 'f', 'n']
case_tags = ['n', 'g', 'd', 'a', 'v']
degree_tags = ['p', 'c', 's']

total_tokens = 0

# Search through every work in the annotated Greek folder
for file in indir[5:]:
    if file[-4:] == '.xml':
        file_count += 1
        print(file_count, file)
        py_samples = []
        pos_labels = []
        person_labels = []
        number_labels = []
        tense_labels = []
        mood_labels = []
        voice_labels = []
        gender_labels = []
        case_labels = []
        degree_labels = []
        dnn_assessing = []
        pos_count = 0
        pos_right = 0
        person_right = 0
        number_right = 0
        tense_right = 0
        mood_right = 0
        voice_right = 0
        gender_right = 0
        case_right = 0
        degree_right = 0
        # Open the files (they are XML's) with beautiful soup and search through every word in every sentence.
        xml_file = open(os.path.join(corpora_folder, file), 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'xml')
        sentences = soup.find_all('sentence')
        for sentence in sentences:
            tokens = sentence.find_all(['word', 'token'])
            for token in tokens:
                if token.has_attr('form') and token.has_attr('postag'):
                    blank_character_tensor = np.array([0]*219, dtype=np.bool_)
                    token_tensor = np.array([blank_character_tensor]*21, dtype=np.bool_)
                    token_length = len(token['form'])
                    for i, character in enumerate(token['form']):
                        character_tensor = np.array([0]*219, dtype=np.bool_)
                        try:
                            character_tensor[all_characters.index(character)] = 1
                        except ValueError:
                            character_tensor[218] = 1
                        token_tensor[21-token_length+i] = character_tensor
                    py_samples.append(token_tensor)
                    try:
                        pos_labels.append(token['postag'][0])
                    except IndexError:
                        pos_labels.append('-')
                    try:
                        person_labels.append(token['postag'][1])
                    except IndexError:
                        person_labels.append('-')
                    try:
                        number_labels.append(token['postag'][2])
                    except IndexError:
                        number_labels.append('-')
                    try:
                        tense_labels.append(token['postag'][3])
                    except IndexError:
                        tense_labels.append('-')
                    try:
                        mood_labels.append(token['postag'][4])
                    except IndexError:
                        mood_labels.append('-')
                    try:
                        voice_labels.append(token['postag'][5])
                    except IndexError:
                        voice_labels.append('-')
                    try:
                        gender_labels.append(token['postag'][6])
                    except IndexError:
                        gender_labels.append('-')
                    try:
                        case_labels.append(token['postag'][7])
                    except IndexError:
                        case_labels.append('-')
                    try:
                        degree_labels.append(token['postag'][8])
                    except IndexError:
                        degree_labels.append('-')
        samples = np.array(py_samples, dtype=np.bool_)
        print('Samples complete... Predicting...')
        predicted_pos_tensors = pos_model.predict(samples)
        predicted_person_tensors = person_model.predict(samples)
        predicted_number_tensors = number_model.predict(samples)
        predicted_tense_tensors = tense_model.predict(samples)
        predicted_mood_tensors = mood_model.predict(samples)
        predicted_voice_tensors = voice_model.predict(samples)
        predicted_gender_tensors = gender_model.predict(samples)
        predicted_case_tensors = case_model.predict(samples)
        predicted_degree_tensors = degree_model.predict(samples)

        i = 0
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
        i = 0
        while i < len(predicted_pos_tensors):
            try:
                predicted_pos = pos_tags[int(np.argmax(predicted_pos_tensors[i]))]
            except IndexError:
                predicted_pos = '-'
            try:
                predicted_person = person_tags[int(np.argmax(predicted_person_tensors[i]))]
            except IndexError:
                predicted_person = '-'
            try:
                predicted_number = number_tags[int(np.argmax(predicted_number_tensors[i]))]
            except IndexError:
                predicted_number = '-'
            try:
                predicted_tense = tense_tags[int(np.argmax(predicted_tense_tensors[i]))]
            except IndexError:
                predicted_tense = '-'
            try:
                predicted_mood = mood_tags[int(np.argmax(predicted_mood_tensors[i]))]
            except IndexError:
                predicted_mood = '-'
            try:
                predicted_voice = voice_tags[int(np.argmax(predicted_voice_tensors[i]))]
            except IndexError:
                predicted_voice = '-'
            try:
                predicted_gender = gender_tags[int(np.argmax(predicted_gender_tensors[i]))]
            except IndexError:
                predicted_gender = '-'
            try:
                predicted_case = case_tags[int(np.argmax(predicted_case_tensors[i]))]
            except IndexError:
                predicted_case = '-'
            try:
                predicted_degree = degree_tags[int(np.argmax(predicted_degree_tensors[i]))]
            except IndexError:
                predicted_degree = '-'
            pos_count += 1
            if predicted_pos == pos_labels[i]:
                pos_right += 1
            if predicted_person == person_labels[i]:
                person_right += 1
            if predicted_number == number_labels[i]:
                number_right += 1
            if predicted_tense == tense_labels[i]:
                tense_right += 1
            if predicted_mood == mood_labels[i]:
                mood_right += 1
            if predicted_voice == voice_labels[i]:
                voice_right += 1
            if predicted_gender == gender_labels[i]:
                gender_right += 1
            if predicted_case == case_labels[i]:
                case_right += 1
            if predicted_degree == degree_labels[i]:
                degree_right += 1
            i += 1
        pos_accuracy = pos_right/pos_count
        person_accuracy = person_right / pos_count
        number_accuracy = number_right/pos_count
        tense_accuracy = tense_right / pos_count
        mood_accuracy = mood_right / pos_count
        voice_accuracy = voice_right / pos_count
        gender_accuracy = gender_right / pos_count
        case_accuracy = case_right / pos_count
        degree_accuracy = degree_right / pos_count
        accuracy_dict[file] = {'POS': pos_accuracy, 'Person': person_accuracy, 'Number': number_accuracy,
                               'Tense': tense_accuracy, 'Mood': mood_accuracy, 'Voice': voice_accuracy,
                               'Gender': gender_accuracy, 'Case': case_accuracy, 'Degree': degree_accuracy}
        print(f'Total samples: {pos_count}')
        print(f'POS Accuracy: {pos_accuracy:.02%}')
        print(f'Person Accuracy: {person_accuracy:.02%}')
        print(f'Number Accuracy: {number_accuracy:.02%}')
        print(f'Tense Accuracy: {tense_accuracy:.02%}')
        print(f'Mood Accuracy: {mood_accuracy:.02%}')
        print(f'Voice Accuracy: {voice_accuracy:.02%}')
        print(f'Gender Accuracy: {gender_accuracy:.02%}')
        print(f'Case Accuracy: {case_accuracy:.02%}')
        print(f'Degree Accuracy: {degree_accuracy:.02%}')

with open('accuracy_records.json', 'w') as outfile:
    json.dump(accuracy_dict, outfile)
