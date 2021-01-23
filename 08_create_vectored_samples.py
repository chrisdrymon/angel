import os
from bs4 import BeautifulSoup
import pickle
import numpy as np
from utilities_morph import elision_normalize
from greek_normalisation.normalise import Normaliser, Norm
from gensim.models import KeyedVectors
import time

agdt_folder = os.path.join('data', 'corpora', 'greek', 'annotated', 'perseus-771dca2', 'texts')
gorman_folder = os.path.join('data', 'corpora', 'greek', 'annotated', 'gorman')
all_files = []
file_count = 0
vectors = []
blank_vector = np.array([0]*100)

# This is just a string that is used in the filename to be saved. Match it with the sorted list indices.
corpus_set = 'AGDT-first26'
for file in sorted(os.listdir(agdt_folder))[:26]:
    all_files.append(os.path.join(agdt_folder, file))
# for file in sorted(os.listdir(gorman_folder)):
#     all_files.append(os.path.join(gorman_folder, file))

# Loading vector dictionary
print('Loading word vector lookups...')
wv = KeyedVectors.load('models/word2vec.wordvectors')

# Create normalizer
normalise = Normaliser().normalise

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
                    wordform = token['form']
                    period_tensor = [0]

                    # Create a feature for periods
                    if wordform == '.':
                        period_tensor = [1]

                    # Look up word embeddings for the wordform
                    try:
                        full_vec = np.concatenate((wv[normalise(elision_normalize(token['form']))[0]], period_tensor),
                                                  axis=0)
                    except KeyError:
                        full_vec = np.concatenate((blank_vector, period_tensor), axis=0)
                    vectors.append(full_vec)
np_vectors = np.array(vectors)
with open(os.path.join('data', 'pickles', f'vectors-{corpus_set}-periods.pickle'), 'wb') as outfile:
    pickle.dump(np_vectors, outfile)
print(np_vectors.shape)
