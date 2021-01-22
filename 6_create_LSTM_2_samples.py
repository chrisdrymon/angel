import os
import time
import json
import pickle
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from utilities_morph import create_morph_classes, ModelSaver

# Make sure the new DNN's are being loaded in the class creations
# Load the trained DNN's
print('Creating morphology classes...')
pos, person, number, tense, mood, voice, gender, case, degree = create_morph_classes()
morphs = (pos, person, number, tense, mood, voice, gender, case, degree)

# Load DNN samples

# Run samples through each DNN
# aspect.dnn_output = aspect.dnn.predict(samples)

# Save as output-DNN-{pos.title}-{corpus_string}.pickle

# If the program is stable, concatenate those outputs. That concatenated tensor represents a single word. Do not
# concatenate the annotator yet.

# Reopen all treebanks. Align