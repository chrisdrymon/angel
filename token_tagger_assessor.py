import os
from bs4 import BeautifulSoup
import time
import json
import datetime
import tensorflow as tf
import numpy as np

corpora_folder = os.path.join('corpora', 'greek', 'annotated', 'perseus-771dca2', 'texts')
models_folder = os.path.join('models')
indir = os.listdir(corpora_folder)
file_count = 0
py_samples = []
py_labels = []
pos_model = tf.keras.models.load_model(os.path.join('models', 'pos-m1x64-0.930val0.899'))
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
