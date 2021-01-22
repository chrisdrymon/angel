import os
import time
import json
import pickle
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from utilities_morph import create_morph_classes, ModelSaver

corpus_string = 'AGDT-last7'

# Load the trained DNN's
print('Creating morphology classes...')
pos, person, number, tense, mood, voice, gender, case, degree = create_morph_classes()
morphs = (pos, person, number, tense, mood, voice, gender, case, degree)

# Load annotator tensors
print('Loading annotator tensors...')
with open(os.path.join('data', 'pickles', f'annotators-{corpus_string}.pickle'), 'rb') as outfile:
    annotator_tensors = pickle.load(outfile)

# Load DNN samples
print('Loading DNN inputs...')
with open(os.path.join('data', 'pickles', f'samples-DNN-AGDT-{corpus_string}.pickle'), 'rb') as infile:
    dnn_input = pickle.load(infile)

# Run samples through each DNN
for aspect in morphs:
    aspect.dnn_output = aspect.dnn.predict(dnn_input)
    with open(os.path.join('data', 'pickles', f'output-DNN-{aspect.title}-{corpus_string}.pickle'), 'wb') as outfile:
        pickle.dump(aspect.dnn_output, outfile)

# If the program is stable, concatenate those outputs. That concatenated tensor represents a single word.
i = 0
lstm2_samples = []
print('Concatenating outputs of the DNN and annotator data...')
while i < len(pos.lstm1_output):
    one_sample = np.concatenate((pos.lstm1_output[i], person.lstm1_output[i], number.lstm1_output[i],
                                 tense.lstm1_output[i], mood.lstm1_output[i], voice.lstm1_output[i],
                                 gender.lstm1_output[i], case.lstm1_output[i], degree.lstm1_output[i],
                                 annotator_tensors[i]), axis=0)
    dnn_samples.append(one_sample)
    i += 1
# Reopen all treebanks. Align