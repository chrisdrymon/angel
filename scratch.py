import os
from bs4 import BeautifulSoup
from collections import Counter
import json
import time
import pandas as pd
from utilities_morph import return_file_annotators

agdt_folder = os.path.join('data', 'corpora', 'greek', 'annotated', 'perseus-771dca2', 'texts')
gorman_folder = os.path.join('data', 'corpora', 'greek', 'annotated', 'gorman')
ignore_names = ['arethusa']
all_files = []

for file in os.listdir(agdt_folder):
    all_files.append(os.path.join(agdt_folder, file))
for file in os.listdir(gorman_folder):
    all_files.append(os.path.join(gorman_folder, file))
with open(os.path.join('data', 'jsons', 'short_annotators.json'), encoding='utf-8') as json_file:
    short_annotators = json.load(json_file)

file_count = 0
for file in all_files[11:]:
    if file[-4:] == '.xml':
        file_count += 1
        print(file_count, file)
        xml_file = open(file, 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'xml')
        sentences = soup.find_all('sentence')
        for sentence in sentences:
            sentence_annotators = []
            xml_sen_ann = sentence.find_all(['annotator', 'primary', 'secondary'])
            for annotator in xml_sen_ann:
                try:
                    sentence_annotators.append(short_annotators[annotator.text])
                except KeyError:
                    sentence_annotators.append(annotator.text)
            print(sentence_annotators)
            tokens = sentence.find_all('token', 'word')
            time.sleep(1)
