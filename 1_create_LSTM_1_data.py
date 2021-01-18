import os
from bs4 import BeautifulSoup
import time
import json
import numpy as np
from utilities_morph import return_sentence_annotators, return_file_annotators
import greek_normalisation

corpora_folder = os.path.join('data', 'corpora', 'greek', 'annotated', 'perseus-771dca2', 'texts')
indir = os.listdir(corpora_folder)
file_count = 0
py_samples = []
py_labels = []

# Load character list and annotator list
with open(os.path.join('data', 'jsons', 'all_characters.json'), encoding='utf-8') as json_file:
    all_characters = json.load(json_file)
with open(os.path.join('data', 'jsons', 'annotators.json'), encoding='utf-8') as json_file:
    annotators = json.load(json_file)
with open(os.path.join('data', 'jsons', 'short_annotators.json'), encoding='utf-8') as json_file:
    short_annotators = json.load(json_file)

pos_tags = ['l', 'n', 'a', 'r', 'c', 'i', 'p', 'v', 'd', 'm', 'g', 'u']
person_tags = ['1', '2', '3']
number_tags = ['s', 'p', 'd']
tense_tags = ['p', 'i', 'r', 'l', 't', 'f', 'a']
mood_tags = ['i', 's', 'n', 'm', 'p', 'o']
voice_tags = ['a', 'p', 'm', 'e']
gender_tags = ['m', 'f', 'n']
case_tags = ['n', 'g', 'd', 'a', 'v']
degree_tags = ['p', 'c', 's']

# Change this for each aspect of morphology
relevant_tagset = pos_tags

# Search through every work in the annotated Greek folder
for file in indir[:26]:
    if file[-4:] == '.xml':
        file_count += 1
        print(file_count, file)

        # Open the files (they are XML's) with beautiful soup and search through every word in every sentence.
        xml_file = open(os.path.join(corpora_folder, file), 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'xml')
        work_annotator = return_file_annotators(soup)
        sentences = soup.find_all('sentence')
        for sentence in sentences:
            sentence_annotators = return_sentence_annotators(sentence, short_annotators)
            tokens = sentence.find_all(['word', 'token'])
            for token in tokens:

                # Enable this if the search should ignore elliptical tokens
                # if token.has_attr('artificial') is False and token.has_attr('empty-token-sort') is False:
                # The longest token is 21 characters. 218 unique characters occur in the unnormalized AGDT corpus.
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

                    # Now create the label tensors.
                    # Go change save file's name right now.
                    # For each aspect of morphology, refactor this tensor's name.
                    pos_tensor = [0] * (len(relevant_tagset) + 1)
                    try:
                        # For each aspect of morphology, change the postag position.
                        pos_tensor[relevant_tagset.index(token['postag'][0])] = 1
                    except IndexError:
                        print(sentence['id'], token['id'], token['form'])
                        pos_tensor[-1] = 1
                    except ValueError:
                        print(sentence['id'], token['id'], token['form'])
                        pos_tensor[-1] = 1
                    py_labels.append(pos_tensor)
samples = np.array(py_samples, dtype=np.bool_)
labels = np.array(py_labels, dtype=np.bool_)
print(f'Samples: {len(samples)}')
print(f'Labels: {len(labels)}')

print('Converting to Numpy Arrays. This may take a few minutes...')
# Split data into an 80%/20% training/validation split.
split = int(.8*len(labels))
train_data = np.array(samples[:split], dtype=np.bool_)
val_data = np.array(samples[split:], dtype=np.bool_)
train_labels = np.array(labels[:split], dtype=np.bool_)
val_labels = np.array(labels[split:], dtype=np.bool_)