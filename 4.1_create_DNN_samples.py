import os
import tensorflow as tf
import pickle
import numpy as np
from utilities_morph import create_morph_classes, ModelSaver

# I've had to split this into two parts because of memory issues.
dnn_samples = []

# Load the datasets
print('Loading datasets...')
with open(os.path.join('data', 'pickles', 'samples-AGDT-first26.pickle'), 'rb') as outfile:
    first26samples = pickle.load(outfile)

# Create the morphology classes
# Check the class. Make sure correct LSTM models are loaded.
print('Loading models...')
pos, person, number, tense, mood, voice, gender, case, degree = create_morph_classes()
morphs = (pos, person, number, tense, mood, voice, gender, case, degree)

# Record and save prediction outputs for each aspect of morphology
for aspect in morphs:
    print(f'Creating outputs from the {aspect.title} model...')
    aspect.lstm1_output = aspect.lstm1.predict(first26samples)
    print(aspect.lstm1_output.shape)
    file_name = f'output-LSTM1-{aspect.title}-AGDT-first26.npy'
    with open(os.path.join('data', 'npys', file_name), 'wb') as outfile:
        np.save(outfile, aspect.lstm1_output)
    print(file_name)
