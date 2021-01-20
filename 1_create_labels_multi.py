import os
from bs4 import BeautifulSoup
import pickle
import json
import numpy as np
from utilities_morph import create_morph_classes

agdt_folder = os.path.join('data', 'corpora', 'greek', 'annotated', 'perseus-771dca2', 'texts')
gorman_folder = os.path.join('data', 'corpora', 'greek', 'annotated', 'gorman')
all_files = []
for file in sorted(os.listdir(agdt_folder))[:26]:
    all_files.append(os.path.join(agdt_folder, file))
for file in sorted(os.listdir(gorman_folder)):
    all_files.append(os.path.join(gorman_folder, file))

# Create morphology aspect classes to simplify tensor sizing and file naming. Keep them in this order.
pos, person, number, tense, mood, voice, gender, case, degree = create_morph_classes()
morphs = (pos, person, number, tense, mood, voice, gender, case, degree)

# This is just a string that is used in the filename to be saved.
corpus_set = 'first26-gorman'

for relevant_morph in morphs[5:]:
    print(f'Creating {relevant_morph.title} labels...')

    # Reset the count and labels list for each aspect of morphology
    file_count = 0
    py_labels = []

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
                        morph_aspect_tensor = [0] * (len(relevant_morph.tags) + 1)
                        try:
                            morph_aspect_tensor[relevant_morph.tags.index(token['postag']
                                                                          [morphs.index(relevant_morph)])] = 1
                        except IndexError:
                            print(sentence['id'], token['id'], token['form'])
                            morph_aspect_tensor[-1] = 1
                        except ValueError:
                            print(sentence['id'], token['id'], token['form'])
                            morph_aspect_tensor[-1] = 1
                        py_labels.append(morph_aspect_tensor)

    print(f'Labels: {len(py_labels)}')
    with open(os.path.join('data', 'pickles', f'labels-{relevant_morph.title}-{corpus_set}-tensors.pickle'),
              'wb') as outfile:
        pickle.dump(py_labels, outfile)
