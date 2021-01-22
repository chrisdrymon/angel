import os
from bs4 import BeautifulSoup
import pickle
import json
import numpy as np
from utilities_morph import return_sentence_annotators, return_file_annotators

agdt_folder = os.path.join('data', 'corpora', 'greek', 'annotated', 'perseus-771dca2', 'texts')
gorman_folder = os.path.join('data', 'corpora', 'greek', 'annotated', 'gorman')
all_files = []
for file in sorted(os.listdir(agdt_folder))[:26]:
    all_files.append(os.path.join(agdt_folder, file))
for file in sorted(os.listdir(gorman_folder)):
    all_files.append(os.path.join(gorman_folder, file))

file_count = 0
py_samples = []
annotator_samples = []

# Load character list and annotator list
with open(os.path.join('data', 'jsons', 'annotators.json'), encoding='utf-8') as json_file:
    all_annotators = json.load(json_file)
with open(os.path.join('data', 'jsons', 'short_annotators.json'), encoding='utf-8') as json_file:
    short_annotators = json.load(json_file)

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
                if token.has_attr('form') and token.has_attr('postag') and token.has_attr('artificial') is False:
                    annotator_tensor = np.array(annotator_tensor, dtype=np.bool_)
                    annotator_samples.append(annotator_tensor)
print(f'Annotator samples: {len(annotator_samples)}')
annotator_samples = np.array(annotator_samples, dtype=np.bool_)
print(f'Numpy Annotator Dimensions: {annotator_samples.shape}')

with open(os.path.join('data', 'pickles', 'annotators-AGDT-first26-gorman-tensors.pickle'), 'wb') as outfile:
    pickle.dump(annotator_samples, outfile)
