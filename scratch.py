import os
from bs4 import BeautifulSoup
from collections import Counter
import json
import time
from utilities_morph import elision_normalize
from greek_normalisation.normalise import Normaliser, Norm

agdt_folder = os.path.join('data', 'corpora', 'greek', 'annotated', 'perseus-771dca2', 'texts')
gorman_folder = os.path.join('data', 'corpora', 'greek', 'annotated', 'gorman')
ignore_names = ['arethusa']
longest = 0
all_files = []
all_characters = []
character_counter = Counter()

for file in os.listdir(agdt_folder):
    all_files.append(os.path.join(agdt_folder, file))
for file in os.listdir(gorman_folder):
    all_files.append(os.path.join(gorman_folder, file))
with open(os.path.join('data', 'jsons', 'short_annotators.json'), encoding='utf-8') as json_file:
    short_annotators = json.load(json_file)
with open(os.path.join('data', 'jsons', 'annotators.json'), encoding='utf-8') as json_file:
    all_annotators = json.load(json_file)

# Create the normaliser
normalise = Normaliser().normalise

file_count = 0
for file in all_files[11:]:
    if file[-4:] == '.xml':
        file_count += 1
        print(file_count, file)
        xml_file = open(file, 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'xml')
        sentences = soup.find_all('sentence')
        for sentence in sentences:
            tokens = sentence.find_all(['word', 'token'])
            for token in tokens:
                if token.has_attr('form') and token.has_attr('postag') and token.has_attr('artificial') is False:
                    normalized_form = normalise(elision_normalize(token['form']))[0]
                    if len(normalized_form) > longest:
                        print(normalized_form, len(normalized_form))
                        longest = len(normalized_form)
print(longest)
