import os
import tensorflow as tf
import pickle
import numpy as np
from utilities_morph import create_morph_classes, ModelSaver

dnn_samples = []

# Load the datasets
print('Loading datasets...')
with open(os.path.join('data', 'pickles', 'samples-AGDT-first26.pickle'), 'rb') as outfile:
    first26samples = pickle.load(outfile)
with open(os.path.join('data', 'pickles', 'annotators-AGDT-first26-tensors.pickle'), 'rb') as outfile:
    annotator_tensors = pickle.load(outfile)

# Create the morphology classes
# Check the class. Make sure correct LSTM models are loaded.
print('Loading models...')
pos, person, number, tense, mood, voice, gender, case, degree = create_morph_classes()
morphs = (pos, person, number, tense, mood, voice, gender, case, degree)

# Record prediction outputs for each aspect of morphology
for aspect in morphs:
    print(f'Creating outputs from the {aspect.title} model...')
    aspect.lstm1_output = aspect.lstm1.predict(first26samples)
    print(aspect.lstm1_output.shape)

i = 0

# # Figure this out. Is the lstm output a list of a numpy array? Figure out how to concatenate correctly. Should I
# # include annotator information in this layer? I think so!
# while i < len(pos.lstm1_output):
#     one_sample = np.concatenate((pos.lstm1_output[i], person.lstm1_output[i], ..., annotator_tensors[i]), axis=0)
#     dnn_samples.append(one_sample)
#
#     i += 1
#
# with open()
