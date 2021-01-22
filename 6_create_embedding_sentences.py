import json
import time
import os
import pickle
from cltk.tokenize.greek.sentence import SentenceTokenizer
from greek_normalisation.normalise import Normaliser, Norm
from utilities_morph import elision_normalize

text_folder = os.path.join('data', 'corpora', 'greek', 'plaintext')

# Create the sentence tokenizer
sent_tokenizer = SentenceTokenizer()

# Create the normalizer
normalise = Normaliser().normalise

all_sentences = []
file_count = 1
for file in sorted(os.listdir(text_folder))[2:]:
    print(file_count, file)
    with open(os.path.join(text_folder, file), 'r', encoding='utf-8') as infile:
        current_text = infile.read()
        normalized_form = normalise(elision_normalize(current_text))[0]
        for greek_sentence in sent_tokenizer.tokenize(normalized_form):
            all_sentences.append(greek_sentence.split())
    print(f'Total sentences: {len(all_sentences)}')
    file_count += 1

with open(os.path.join('data', 'jsons', 'greek_sentences.json'), 'w', encoding='utf-8') as outfile:
    json.dump(all_sentences, outfile, ensure_ascii=False)
