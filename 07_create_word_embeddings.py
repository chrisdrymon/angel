from gensim.models import Word2Vec
import json
import os

with open(os.path.join('data', 'jsons', 'tokenized_sentences.json'), 'r', encoding='utf-8') as infile:
    greek_sentences = json.load(infile)

model = Word2Vec(sentences=greek_sentences, size=100, window=10, min_count=5, workers=4, iter=5)
model.save(os.path.join('models', 'word2vec.model'))
word_vectors = model.wv
word_vectors.save(os.path.join('models', 'word2vec.wordvectors'))