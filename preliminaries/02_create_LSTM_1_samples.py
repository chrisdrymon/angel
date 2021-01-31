import os
from bs4 import BeautifulSoup
import json
import pickle
import numpy as np
from preliminaries.utilities_morph import return_sentence_annotators, return_file_annotators, elision_normalize
from greek_normalisation.normalise import Normaliser

agdt_folder = os.path.join('../data', 'corpora', 'greek', 'annotated', 'perseus-771dca2', 'texts')
gorman_folder = os.path.join('../data', 'corpora', 'greek', 'annotated', 'gorman')
all_files = []
for file in sorted(os.listdir(agdt_folder))[-7:]:
    all_files.append(os.path.join(agdt_folder, file))
# for file in sorted(os.listdir(gorman_folder)):
#     all_files.append(os.path.join(gorman_folder, file))

file_count = 0
py_samples = []

# Load character list and annotator list
with open(os.path.join('../data', 'jsons', 'all_norm_characters.json'), encoding='utf-8') as json_file:
    all_norm_characters = json.load(json_file)
with open(os.path.join('../data', 'jsons', 'annotators.json'), encoding='utf-8') as json_file:
    all_annotators = json.load(json_file)
with open(os.path.join('../data', 'jsons', 'short_annotators.json'), encoding='utf-8') as json_file:
    short_annotators = json.load(json_file)

# Create the normalizer
normalise = Normaliser().normalise

# Search through every work in the annotated Greek folder
for file in all_files:
    if file[-4:] == '.xml':
        file_count += 1
        print(file_count, file)

        # Open the files (they are XML's) with beautiful soup and search through every word in every sentence.
        xml_file = open(file, 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'xml')
        work_annotators = return_file_annotators(soup)
        sentences = soup.find_all('sentence')
        for sentence in sentences:

            # Prepare annotator tensor. There are 36 annotators for Gorman/AGDT.
            sentence_annotators = return_sentence_annotators(sentence, short_annotators)
            if not sentence_annotators:
                sentence_annotators = work_annotators
            annotator_tensor = [0]*37
            for anno in sentence_annotators:
                try:
                    annotator_tensor[all_annotators.index(anno)] = 1
                except IndexError:
                    annotator_tensor[-1] = 1

            tokens = sentence.find_all(['word', 'token'])
            for token in tokens:

                # In the unnormalized AGDT corpus, there are 218 unique characters. The longest token is 21 characters.
                # In the normalized/elision-normalized Gorman-AGDT corpus, there are 136 unique characters. Its longest
                # token is 22 characters. Wordform size will be capped at 21 character. Any longer than that will keep
                # the first 10 and last 10 characters. The center will be marked with an elision "character". One
                # character which occurred once in the corpus was erased from the character list and replaced as an
                # "other" character. 135 characters + 1 elision + 1 other = 137-length tensor.
                if token.has_attr('form') and token.has_attr('postag') and token.has_attr('artificial') is False:
                    blank_character_tensor = np.array([0]*174, dtype=np.bool_)

                    # The whole token tensor start out blank because it's challenging to fill out the empty characters.
                    token_tensor = np.array([blank_character_tensor]*21, dtype=np.bool_)

                    # Normalize each token before tensorizing its characters.
                    normalized_form = normalise(elision_normalize(token['form']))[0]
                    token_length = len(normalized_form)

                    # Create token tensors for tokens longer than 21 characters
                    if token_length > 21:
                        token_tensor = []
                        for character in normalized_form[:10]:
                            character_tensor = [0]*137
                            try:
                                character_tensor[all_norm_characters.index(character)] = 1
                            except ValueError:
                                character_tensor[136] = 1

                            # Append the annotator tensor at the end of every character tensor
                            character_tensor = character_tensor + annotator_tensor
                            character_tensor = np.array(character_tensor, dtype=np.bool_)
                            token_tensor.append(character_tensor)
                        character_tensor = [0]*137
                        character_tensor[135] = 1

                        # Append the annotator tensor at the end of every character tensor
                        character_tensor = character_tensor + annotator_tensor
                        character_tensor = np.array(character_tensor, dtype=np.bool_)
                        token_tensor.append(character_tensor)
                        for character in normalized_form[-10:]:
                            character_tensor = [0]*137
                            try:
                                character_tensor[all_norm_characters.index(character)] = 1
                            except ValueError:
                                character_tensor[136] = 1

                            # Append the annotator tensor at the end of every character tensor
                            character_tensor = character_tensor + annotator_tensor
                            character_tensor = np.array(character_tensor, dtype=np.bool_)
                            token_tensor.append(character_tensor)
                        token_tensor = np.array(token_tensor, dtype=np.bool_)

                    # Create token tensors for tokens shorter than 22 characters
                    else:
                        for i, character in enumerate(normalized_form):
                            character_tensor = [0]*137
                            try:
                                character_tensor[all_norm_characters.index(character)] = 1
                            except ValueError:
                                character_tensor[136] = 1

                            # Append the annotator tensor at the end of every character tensor
                            character_tensor = character_tensor + annotator_tensor
                            character_tensor = np.array(character_tensor)
                            token_tensor[21-token_length+i] = character_tensor

                    # Add each tensor token to the samples
                    py_samples.append(token_tensor)
samples = np.array(py_samples, dtype=np.bool_)
with open(os.path.join('../data', 'pickles', 'samples.pickle'), 'wb') as outfile:
    pickle.dump(samples, outfile)
