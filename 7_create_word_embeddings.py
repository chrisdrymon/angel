from gensim.models import Word2Vec
import json
import os

with open(os.path.join('data', 'jsons', 'text_sentences.json'), 'r', encoding='utf-8') as infile:
    greek_sentences = json.load(infile)

model = Word2Vec(sentences=greek_sentences[:100], size=100, window=5, min_count=1, workers=4)
print(f'καί in vector form is {model.wv["καί"]}')
print(f'καιροῖς in vector form is {model.wv["καιροῖς"]}')
model.save(os.path.join('models', 'word2vec.model'))
