import numpy as np
import os
from preliminaries.utilities_morph import create_morph_classes
import pickle

pos, person, number, tense, mood, voice, gender, case, degree = create_morph_classes()
morphs = (pos, person, number, tense, mood, voice, gender, case, degree)

corpus_string = 'AGDT-first26'

with open(os.path.join('../data', 'pickles', f'annotators-{corpus_string}.pickle'), 'rb') as outfile:
    annotator_tensors = pickle.load(outfile)
for aspect in morphs:
    filename = f'output-LSTM1-{aspect.title}-{corpus_string}.pickle'
    with open(os.path.join('../data', 'pickles', filename), 'rb') as outfile:
        aspect.lstm1_output = pickle.load(outfile)
    print(f'Opened {filename} with shape {aspect.lstm1_output.shape}')

dnn_samples = []
i = 0

print('Concatenating outputs of LSTM1 and annotator data...')
while i < len(pos.lstm1_output):
    one_sample = np.concatenate((pos.lstm1_output[i], person.lstm1_output[i], number.lstm1_output[i],
                                 tense.lstm1_output[i], mood.lstm1_output[i], voice.lstm1_output[i],
                                 gender.lstm1_output[i], case.lstm1_output[i], degree.lstm1_output[i],
                                 annotator_tensors[i]), axis=0)
    dnn_samples.append(one_sample)
    i += 1

dnn_samples = np.array(dnn_samples)
print(f'DNN Samples shape: {dnn_samples.shape}')
new_file = f'samples-DNN-{corpus_string}.pickle'
with open(os.path.join('../data', 'pickles', new_file), 'wb') as outfile:
    pickle.dump(dnn_samples, outfile)
print(f'Wrote {new_file} to disk.')
