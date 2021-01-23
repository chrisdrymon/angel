import os
from bs4 import BeautifulSoup
from collections import Counter
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from utilities_morph import elision_normalize
from greek_normalisation.normalise import Normaliser, Norm
import tensorflow as tf

with open(os.path.join('data', 'pickles', 'samples-LSTM2-AGDT-last7.pickle'), 'rb') as infile1:
    val_data = pickle.load(infile1)

with open(os.path.join('data', 'pickles', 'labels-pos-AGDT-last7.pickle'), 'rb') as infile2:
    val_labels = pickle.load(infile2)

print(val_data.shape)
print(val_labels.shape)
data_blank_array = np.array([0]*192)
label_blank_array = np.array([0]*13)
start_data_padding = np.tile(data_blank_array, (7, 1))
end_data_padding = np.tile(data_blank_array, (8, 1))
start_label_padding = np.tile(label_blank_array, (15, 1))

padded_val_data = np.concatenate((start_data_padding, val_data, end_data_padding), axis=0)
padded_val_labels = np.concatenate((start_label_padding, val_labels), axis=0)

print(padded_val_data.shape)
print(padded_val_labels.shape)

data_gen = TimeseriesGenerator(padded_val_data, padded_val_labels, length=15, sampling_rate=1, batch_size=1)
# Batch, tuple(samples, labels),
print(data_gen[1][0].shape)
print(data_gen[1][1])


