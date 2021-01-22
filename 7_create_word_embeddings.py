from gensim.test.utils import common_texts
from gensim.models import Word2Vec




model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")