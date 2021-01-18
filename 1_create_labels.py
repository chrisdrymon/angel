import os
from bs4 import BeautifulSoup
import pickle
import json
import numpy as np

agdt_folder = os.path.join('data', 'corpora', 'greek', 'annotated', 'perseus-771dca2', 'texts')
gorman_folder = os.path.join('data', 'corpora', 'greek', 'annotated', 'gorman')
all_files = []
for file in sorted(os.listdir(agdt_folder))[:26]:
    all_files.append(os.path.join(agdt_folder, file))
for file in sorted(os.listdir(gorman_folder)):
    all_files.append(os.path.join(gorman_folder, file))

file_count = 0
py_labels = []

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
for file in all_files:
    if file[-4:] == '.xml':
        file_count += 1
        print(file_count, file)

        # Open the files (they are XML's) with beautiful soup and search through every word in every sentence.
        xml_file = open(os.path.join(file), 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'xml')
        sentences = soup.find_all('sentence')
        for sentence in sentences:
            tokens = sentence.find_all(['word', 'token'])
            for token in tokens:
                if token.has_attr('form') and token.has_attr('postag') and token.has_attr('artificial') is False:

                    # Now create the label tensors.
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
# labels = np.array(py_labels, dtype=np.bool_)
print(f'Labels: {len(py_labels)}')
with open(os.path.join('data', 'pickles', 'labels.pickle'), 'wb') as outfile:
    pickle.dump(py_labels, outfile)
