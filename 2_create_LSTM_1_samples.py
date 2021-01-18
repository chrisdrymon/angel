import os
from bs4 import BeautifulSoup
import time
import json
import numpy as np
from utilities_morph import return_sentence_annotators, return_file_annotators, elision_normalize
from greek_normalisation.normalise import Normaliser, Norm

corpora_folder = os.path.join('data', 'corpora', 'greek', 'annotated', 'perseus-771dca2', 'texts')
indir = os.listdir(corpora_folder)
file_count = 0
py_samples = []

# Load character list and annotator list
with open(os.path.join('data', 'jsons', 'all_norm_characters.json'), encoding='utf-8') as json_file:
    all_norm_characters = json.load(json_file)
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

# Create the normalizer
normalise = Normaliser().normalise

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

                # In the unnormalized AGDT corpus, there are 218 unique characters. The longest token is 21 characters.
                # In the normalized/elision-normalized Gorman-AGDT corpus, there are 136 unique characters. Its longest
                # token is 22 characters. Wordform size will be capped at 21 character. Any longer than that will keep
                # the first 10 and last 10 characters. The center will be marked with an elision "character". One
                # character which occurred once in the corpus was erased from the character list and replaced as an
                # "other" character. 135 characters + 1 elision + 1 other = 137-length tensor.
                if token.has_attr('form') and token.has_attr('postag') and token.has_attr('artificial') is False:
                    blank_character_tensor = np.array([0]*137, dtype=np.bool_)
                    token_tensor = np.array([blank_character_tensor]*21, dtype=np.bool_)

                    # Normalize each token before tensorizing its characters.
                    wordform = normalise(elision_normalize(token['form']))

                    token_length = len(token['form'])



                    # Create token tensors for tokens longer than 21 characters
                    if token_length > 21:
                        token_tensor = []
                        for character in token['form'][:10]:
                            character_tensor = np.array([0]*137, dtype=np.bool_)
                            try:
                                character_tensor[all_norm_characters.index(character)] = 1
                            except ValueError:
                                character_tensor[136] = 1
                            token_tensor.append(character_tensor)
                        character_tensor = np.array([0]*137, dtype=np.bool_)
                        character_tensor[135] = 1
                        token_tensor.append(character_tensor)
                        for character in token['form'][-10:]:
                            character_tensor = np.array([0]*137, dtype=np.bool_)
                            try:
                                character_tensor[all_norm_characters.index(character)] = 1
                            except ValueError:
                                character_tensor[136] = 1
                            token_tensor.append(character_tensor)
                        token_tensor = np.array(token_tensor, dtype=np.bool_)

                    # Create token tensors for tokens shorter than 22 characters
                    else:
                        for i, character in enumerate(token['form']):
                            character_tensor = np.array([0]*137, dtype=np.bool_)
                            try:
                                character_tensor[all_norm_characters.index(character)] = 1
                            except ValueError:
                                character_tensor[136] = 1
                            token_tensor[21-token_length+i] = character_tensor
                    py_samples.append(token_tensor)
samples = np.array(py_samples)
