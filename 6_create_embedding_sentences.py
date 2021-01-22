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
        no_blanks = ' '.join(normalized_form.split())
        all_sentences = all_sentences + sent_tokenizer.tokenize(no_blanks)
        print(f'{len(all_sentences)} total sentences.')
    file_count += 1

with open(os.path.join('data', 'jsons', 'text_sentences.json'), 'w', encoding='utf-8') as outfile:
    json.dump(all_sentences, outfile, ensure_ascii=False)
