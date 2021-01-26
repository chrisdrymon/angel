import os
import json
from greek_normalisation.normalise import Normaliser
import numpy as np
import tensorflow as tf
import pickle
from gensim.models import KeyedVectors
from tensorflow.keras.models import load_model


class Morphs:
    """Hold data for one aspect of morphology."""
    def __init__(self, title, tags, lstm1, dnn, lstm2):
        self.title = title
        self.tags = tags
        self.lstm1 = lstm1
        self.lstm2 = lstm2
        self.dnn = dnn
        self.output1 = []
        self.output2 = []
        self.output3 = []
        self.predicted_tags1 = []
        self.predicted_tags2 = []
        self.predicted_tags3 = []
        self.confidence1 = []
        self.confidence2 = []
        self.confidence3 = []


def create_morph_classes():
    """Create a class instance for each part of speech aspect."""
    print('Part-of-speech models loading...')
    pos_lstm1 = load_model(os.path.join('models', 'pos-lstm1-3x128-0.927val0.939-AGDTfirst26last7.h5'))
    pos_dnn = load_model(os.path.join('models', 'pos-dnn-2x20-0.939val0.942-AGDTfirst26last7.h5'))
    pos_lstm2 = load_model(os.path.join('models', 'pos-lstm2-3x128-0.958val0.955-AGDTfirst26last7.h5'))

    print('Person models loading...')
    person_lstm1 = load_model(os.path.join('models', 'person-lstm1-3x128-0.983val0.990-AGDTfirst26last7.h5'))
    person_dnn = load_model(os.path.join('models', 'person-dnn-2x20-0.994val0.992-AGDTfirst26last7.h5'))
    person_lstm2 = load_model(os.path.join('models', 'person-lstm2-3x128-0.994val0.994-AGDTfirst26last7.h5'))

    print('Number models loading...')
    number_lstm1 = load_model(os.path.join('models', 'number-lstm1-3x128-0.955val0.980-AGDTfirst26last7.h5'))
    number_dnn = load_model(os.path.join('models', 'number-dnn-2x20-0.977val0.981-AGDTfirst26last7.h5'))
    number_lstm2 = load_model(os.path.join('models', 'number-lstm2-3x128-0.985val0.987-AGDTfirst26last7.h5'))

    print('Tense models loading...')
    tense_lstm1 = load_model(os.path.join('models', 'tense-lstm1-3x128-0.976val0.990-AGDTfirst26last7.h5'))
    tense_dnn = load_model(os.path.join('models', 'tense-dnn-2x20-0.990val0.992-AGDTfirst26last7.h5'))
    tense_lstm2 = load_model(os.path.join('models', 'tense-lstm2-3x128-0.986val0.992-AGDTfirst26last7.h5'))

    print('Mood models loading...')
    mood_lstm1 = load_model(os.path.join('models', 'mood-lstm1-3x128-0.981val0.992-AGDTfirst26last7.h5'))
    mood_dnn = load_model(os.path.join('models', 'mood-dnn-2x20-0.994val0.992-AGDTfirst26last7.h5'))
    mood_lstm2 = load_model(os.path.join('models', 'mood-lstm2-3x128-0.993val0.994-AGDTfirst26last7.h5'))

    print('Voice models loading...')
    voice_lstm1 = load_model(os.path.join('models', 'voice-lstm1-3x128-0.978val0.991-AGDTfirst26last7.h5'))
    voice_dnn = load_model(os.path.join('models', 'voice-dnn-2x20-0.992val0.993-AGDTfirst26last7.h5'))
    voice_lstm2 = load_model(os.path.join('models', 'voice-lstm2-3x128-'))

    print('Gender models loading...')
    gender_lstm1 = load_model(os.path.join('models', 'gender-lstm1-3x128-0.923val0.934-AGDTfirst26last7.h5'))
    gender_dnn = load_model(os.path.join('models', 'gender-dnn-2x20-0.952val0.937-AGDTfirst26last7.h5'))
    gender_lstm2 = load_model(os.path.join('models', 'gender-lstm2-3x128-'))

    print('Case models loading...')
    case_lstm1 = load_model(os.path.join('models', 'case-lstm1-3x128-0.934val0.962-AGDTfirst26last7.h5'))
    case_dnn = load_model(os.path.join('models', 'case-dnn-2x20-0.957val0.963-AGDTfirst26last7.h5'))
    case_lstm2 = load_model(os.path.join('models', 'case-lstm2-3x128-'))

    print('Degree models loading...')
    degree_lstm1 = load_model(os.path.join('models', 'degree-lstm1-3x128-0.998val0.999-AGDTfirst26last7.h5'))
    degree_dnn = load_model(os.path.join('models', 'degree-dnn-2x20-0.999val0.999-AGDTfirst26last7.h5'))
    degree_lstm2 = load_model(os.path.join('models', 'degree-lstm2-3x128-'))

    # The possible tags for each item of morphology
    pos_tags = ('l', 'n', 'a', 'r', 'c', 'i', 'p', 'v', 'd', 'm', 'g', 'u')
    person_tags = ('1', '2', '3')
    number_tags = ('s', 'p', 'd')
    tense_tags = ('p', 'i', 'r', 'l', 't', 'f', 'a')
    mood_tags = ('i', 's', 'n', 'm', 'p', 'o')
    voice_tags = ('a', 'p', 'm', 'e')
    gender_tags = ('m', 'f', 'n')
    case_tags = ('n', 'g', 'd', 'a', 'v')
    degree_tags = ('p', 'c', 's')

    # Create a class instance for each aspect of morphology
    fpos = Morphs('pos', pos_tags, pos_lstm1, pos_dnn, pos_lstm2)
    fperson = Morphs('person', person_tags, person_lstm1, person_dnn, person_lstm2)
    fnumber = Morphs('number', number_tags, number_lstm1, number_dnn, number_lstm2)
    ftense = Morphs('tense', tense_tags, tense_lstm1, tense_dnn, tense_lstm2)
    fmood = Morphs('mood', mood_tags, mood_lstm1, mood_dnn, mood_lstm2)
    fvoice = Morphs('voice', voice_tags, voice_lstm1, voice_dnn, voice_lstm2)
    fgender = Morphs('gender', gender_tags, gender_lstm1, gender_dnn, gender_lstm2)
    fcase = Morphs('case', case_tags, case_lstm1, case_dnn, case_lstm2)
    fdegree = Morphs('degree', degree_tags, degree_lstm1, degree_dnn, degree_lstm2)

    return fpos, fperson, fnumber, ftense, fmood, fvoice, fgender, fcase, fdegree


def elision_normalize(s):
    """Turn unicode characters which look similar to 2019 into 2019."""
    return s.replace("\u02BC", "\u2019").replace("\u1FBF", "\u2019").replace("\u0027", "\u2019").\
        replace("\u1FBD", "\u2019")


def isolate_greek_punctuation(fsentence):
    return fsentence.replace(',', ' , ').replace('·', ' · ').replace(';', ' ; ').replace('.', ' . ').\
        replace('?', ' ? ').replace('»', ' » ').replace('«', ' « ').replace('“', ' “ ').replace('„', ' „ ')


def remove_greek_punctuation(fword):
    """Return the Greek input without punctuation."""
    return fword.replace(',', '').replace('·', '').replace(';', '').replace('.', '').replace('?', '').replace('»', '').\
        replace('«', '').replace('“', '').replace('„', '')


def vector_lookup(gword):
    try:
        return wv[gword]
    except KeyError:
        return np.array([0]*100)


greek_text = 'νέος μὲν καὶ ἄπειρος δικῶν ἔγωγε ἔτι, δεινῶς δὲ καὶ ἀπόρως ἔχει μοι περὶ τοῦ πράγματος, ὦ ἄνδρες, ' \
             'τοῦτο μὲν εἰ ἐπισκήψαντος τοῦ πατρὸς ἐπεξελθεῖν τοῖς αὑτοῦ φονεῦσι μὴ ἐπέξειμι, τοῦτο δὲ εἰ ἐπεξιόντι ' \
             'ἀναγκαίως ἔχει οἷς ἥκιστα ἐχρῆν ἐν διαφορᾷ καταστῆναι, ἀδελφοῖς ὁμοπατρίοις καὶ μητρὶ ἀδελφῶν.'
annotator = 'Vanessa Gorman'

# Add LSTM2 to these when they are ready
print('Loading models...')
pos, person, number, tense, mood, voice, gender, case, degree = create_morph_classes()
morphs = (pos, person, number, tense, mood, voice, gender, case, degree)

# Load character list, annotator list, and vector dictionary
print('Loading character list, annotator list, and vector dictionary...')
with open(os.path.join('data', 'jsons', 'all_norm_characters.json'), encoding='utf-8') as json_file:
    all_norm_characters = json.load(json_file)
with open(os.path.join('data', 'jsons', 'annotators.json'), encoding='utf-8') as json_file:
    all_annotators = json.load(json_file)
with open(os.path.join('data', 'jsons', 'short_annotators.json'), encoding='utf-8') as json_file:
    short_annotators = json.load(json_file)
wv = KeyedVectors.load('models/fasttext.wordvectors')

# Create the normalizer
normalise = Normaliser().normalise

# Create annotator tensor
annotator_tensor = [0] * 37
try:
    annotator_tensor[all_annotators.index(annotator)] = 1

# Make Vanessa Gorman the default annotator
except IndexError:
    annotator_tensor[0] = 1

print('Pre-processing text...')
blank_character_tensor = np.array([0]*174, dtype=np.float32)
punc_separated_text = isolate_greek_punctuation(greek_text)
split_text = punc_separated_text.split()
print(f'Text and punctuation split into {len(split_text)} individual tokens.')
one_hotted_tokens = []
dnn_input = []
blank_lstm2_token = np.array([0]*192)
lstm2_padding = np.tile(blank_lstm2_token, (7, 1))
lstm2_input = []

# Create character tensors and word tensors composed of those character tensors
for word in split_text:

    # The whole token tensor starts out blank because it's challenging to fill out the empty characters.
    token_tensor = np.array([blank_character_tensor]*21, dtype=np.float32)

    # Normalize each token before tensorizing its characters.
    normalized_form = normalise(elision_normalize(word))[0]
    token_length = len(normalized_form)

    # Create token tensors for tokens longer than 21 characters
    if token_length > 21:
        token_tensor = []
        for character in normalized_form[:10]:
            character_tensor = [0]*137
            try:
                character_tensor[all_norm_characters.index(character)] = 1
            except ValueError:
                character_tensor[136] = 1

            # Append the annotator tensor at the end of every character tensor
            character_tensor = character_tensor + annotator_tensor
            character_tensor = np.array(character_tensor, dtype=np.float32)
            token_tensor.append(character_tensor)
        character_tensor = [0]*137
        character_tensor[135] = 1

        # Append the annotator tensor at the end of every character tensor
        character_tensor = character_tensor + annotator_tensor
        character_tensor = np.array(character_tensor, dtype=np.float32)
        token_tensor.append(character_tensor)
        for character in normalized_form[-10:]:
            character_tensor = [0]*137
            try:
                character_tensor[all_norm_characters.index(character)] = 1
            except ValueError:
                character_tensor[136] = 1

            # Append the annotator tensor at the end of every character tensor
            character_tensor = character_tensor + annotator_tensor
            character_tensor = np.array(character_tensor, dtype=np.float32)
            token_tensor.append(character_tensor)
        token_tensor = np.array(token_tensor, dtype=np.float32)

    # Create token tensors for tokens shorter than 22 characters
    else:
        for i, character in enumerate(normalized_form):
            character_tensor = [0]*137
            try:
                character_tensor[all_norm_characters.index(character)] = 1
            except ValueError:
                character_tensor[136] = 1

            # Append the annotator tensor at the end of every character tensor
            character_tensor = character_tensor + annotator_tensor
            character_tensor = np.array(character_tensor, dtype=np.float32)
            token_tensor[21-token_length+i] = character_tensor

    # Add each tensor token to the samples
    one_hotted_tokens.append(token_tensor)
one_hots_np = np.array(one_hotted_tokens, dtype=np.float32)

# Process through the first LSTM...
print('Making LSTM 1 predictions...')
for aspect in morphs:
    aspect.output1 = aspect.lstm1.predict(one_hots_np)

for aspect in morphs:
    for tensor in aspect.output1:
        try:
            aspect.predicted_tags1.append(aspect.tags[int(np.argmax(tensor))])
        except IndexError:
            aspect.predicted_tags1.append('-')
        aspect.confidence1.append(np.amax(tensor))

for i, token in enumerate(punc_separated_text.split()):
    dnn_input.append(np.concatenate((pos.output1[i], person.output1[i], number.output1[i], tense.output1[i],
                                    mood.output1[i], voice.output1[i], gender.output1[i], case.output1[i],
                                    degree.output1[i], annotator_tensor), axis=0))
np_dnn_input = np.array(dnn_input)

# Run outputs through DNN
print('Making DNN predictions...')
for aspect in morphs:
    aspect.output2 = aspect.dnn.predict(np_dnn_input)

for aspect in morphs:
    for tensor in aspect.output2:
        try:
            aspect.predicted_tags2.append(aspect.tags[int(np.argmax(tensor))])
        except IndexError:
            aspect.predicted_tags2.append('-')
        aspect.confidence2.append(np.amax(tensor))

# Prepare inputs for LSTM2
for i, token in enumerate(punc_separated_text.split()):
    lstm2_input.append(np.concatenate((pos.output2[i], person.output2[i], number.output2[i], tense.output2[i],
                                       mood.output2[i], voice.output2[i], gender.output2[i], case.output2[i],
                                       degree.output2[i], annotator_tensor,
                                       vector_lookup(normalise(elision_normalize(token))[0])), axis=0))

padded_lstm2_input = np.concatenate((lstm2_padding, lstm2_input, lstm2_padding))

time_series = []
for i in range(0, len(padded_lstm2_input)-14):
    time_series.append(padded_lstm2_input[i:i+15])

lstm2_ts = np.array(time_series)

# Run outputs through LSTM2
print('Making LSTM 2 predictions...')
for aspect in morphs:
    aspect.output3 = aspect.lstm2.predict(lstm2_ts)

for aspect in morphs:
    for tensor in aspect.output3:
        try:
            aspect.predicted_tags3.append(aspect.tags[int(np.argmax(tensor))])
        except IndexError:
            aspect.predicted_tags3.append('-')
        aspect.confidence3.append(np.amax(tensor))

for i, token in enumerate(punc_separated_text.split()):
    print(token, f'{pos.confidence1[i]:.02%}',
          pos.predicted_tags1[i] + person.predicted_tags1[i] +
          number.predicted_tags1[i] + tense.predicted_tags1[i] + mood.predicted_tags1[i] + voice.predicted_tags1[i] +
          gender.predicted_tags1[i] + case.predicted_tags1[i] + degree.predicted_tags1[i],
          f'{pos.confidence2[i]:.02%}',
          pos.predicted_tags2[i] + person.predicted_tags2[i] +
          number.predicted_tags2[i] + tense.predicted_tags2[i] + mood.predicted_tags2[i] + voice.predicted_tags2[i] +
          gender.predicted_tags2[i] + case.predicted_tags2[i] + degree.predicted_tags2[i],
          f'{pos.confidence3[i]:.02%}',
          pos.predicted_tags3[i] + person.predicted_tags3[i] +
          number.predicted_tags3[i] + tense.predicted_tags3[i] + mood.predicted_tags3[i] + voice.predicted_tags3[i] +
          gender.predicted_tags3[i] + case.predicted_tags3[i] + degree.predicted_tags3[i])
