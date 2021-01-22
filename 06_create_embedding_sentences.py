import json
import time
import os
import pickle
from cltk.tokenize.greek.sentence import SentenceTokenizer
from greek_normalisation.normalise import Normaliser, Norm
from utilities_morph import elision_normalize, remove_greek_punctuation

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

        # Split the text into sentences
        for greek_sentence in sent_tokenizer.tokenize(current_text):
            new_sentence = []

            # For each word in the sentence, remove punctuation and normalise its spelling.
            for word in greek_sentence.split():
                word = remove_greek_punctuation(word)
                normalized_form = normalise(elision_normalize(word))[0]
                new_sentence.append(normalized_form)

            # Each sentence needs to be saved as a list of word.
            all_sentences.append(new_sentence)
    print(f'{len(all_sentences)} total sentences.')
    file_count += 1

with open(os.path.join('data', 'jsons', 'greek_sentences.json'), 'w', encoding='utf-8') as outfile:
    json.dump(all_sentences, outfile, ensure_ascii=False)
