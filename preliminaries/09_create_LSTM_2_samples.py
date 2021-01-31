import os
import pickle
import numpy as np
from preliminaries.utilities_morph import create_morph_classes

# Change this to fit the target corpus: AGDT-first26 or AGDT-last7
corpus_string = 'AGDT-last7'

# Load the trained DNN's
print('Creating morphology classes...')
pos, person, number, tense, mood, voice, gender, case, degree = create_morph_classes()
morphs = (pos, person, number, tense, mood, voice, gender, case, degree)

# Load annotator tensors
print('Loading annotator tensors...')
with open(os.path.join('../data', 'pickles', f'annotators-{corpus_string}.pickle'), 'rb') as infile1:
    annotator_tensors = pickle.load(infile1)

# Load DNN samples
print('Loading DNN inputs...')
with open(os.path.join('../data', 'pickles', f'samples-DNN-{corpus_string}.pickle'), 'rb') as infile2:
    dnn_input = pickle.load(infile2)

# Load word vectors
print('Loading word vectors...')
with open(os.path.join('../data', 'pickles', f'vectors-fasttext-{corpus_string}.pickle'), 'rb') as infile3:
    vector_samples = pickle.load(infile3)

# Run samples through each DNN
for aspect in morphs:
    print(f'Running samples through {aspect.title} DNN...')
    aspect.dnn_output = aspect.dnn.predict(dnn_input)
    with open(os.path.join('../data', 'pickles', f'output-DNN-{aspect.title}-{corpus_string}.pickle'), 'wb') as outfile:
        pickle.dump(aspect.dnn_output, outfile)

# If the program is stable, concatenate those outputs. That concatenated tensor represents a single word.
i = 0
lstm2_samples = []
print('Concatenating outputs of the DNN, annotator data, and word vector...')
while i < len(pos.dnn_output):
    one_sample = np.concatenate((pos.dnn_output[i], person.dnn_output[i], number.dnn_output[i],
                                 tense.dnn_output[i], mood.dnn_output[i], voice.dnn_output[i],
                                 gender.dnn_output[i], case.dnn_output[i], degree.dnn_output[i],
                                 annotator_tensors[i], vector_samples[i]), axis=0)
    lstm2_samples.append(one_sample)
    if i % 10000 == 0:
        print(f'{i}/{len(pos.dnn_output)} complete')
    i += 1

np_samples = np.array(lstm2_samples)
with open(os.path.join('../data', 'pickles', f'samples-LSTM2-fasttext-{corpus_string}.pickle'), 'wb') as outfile:
    pickle.dump(np_samples, outfile)
