import os
import json
from greek_normalisation.normalise import Normaliser
import numpy as np
import tensorflow as tf


class Morphs:
    """Hold data for one aspect of morphology."""
    def __init__(self, title, tags, lstm1, dnn):
        self.title = title
        self.tags = tags
        self.lstm1 = lstm1
        self.dnn = dnn


def create_morph_classes():
    """Create a class instance for each part of speech aspect."""
    pos_lstm1 = tf.keras.models.load_model(os.path.join('models', 'pos-lstm1-3x128-0.927val0.939-AGDTfirst26last7.h5'))
    pos_dnn = tf.keras.models.load_model(os.path.join('models', 'pos-dnn-2x20-0.939val0.942-AGDTfirst26last7.h5'))
    person_lstm1 = tf.keras.models.load_model(os.path.join('models',
                                                           'person-lstm1-3x128-0.983val0.990-AGDTfirst26last7.h5'))
    person_dnn = tf.keras.models.load_model(os.path.join('models', 'person-dnn-2x20-0.994val0.992-AGDTfirst26last7.h5'))
    number_lstm1 = tf.keras.models.load_model(os.path.join('models',
                                                           'number-lstm1-3x128-0.955val0.980-AGDTfirst26last7.h5'))
    number_dnn = tf.keras.models.load_model(os.path.join('models', 'number-dnn-2x20-0.977val0.981-AGDTfirst26last7.h5'))
    tense_lstm1 = tf.keras.models.load_model(os.path.join('models',
                                                          'tense-lstm1-3x128-0.976val0.990-AGDTfirst26last7.h5'))
    tense_dnn = tf.keras.models.load_model(os.path.join('models', 'tense-dnn-2x20-0.990val0.992-AGDTfirst26last7.h5'))
    mood_lstm1 = tf.keras.models.load_model(os.path.join('models',
                                                         'mood-lstm1-3x128-0.981val0.992-AGDTfirst26last7.h5'))
    mood_dnn = tf.keras.models.load_model(os.path.join('models', 'mood-dnn-2x20-0.994val0.992-AGDTfirst26last7.h5'))
    voice_lstm1 = tf.keras.models.load_model(os.path.join('models',
                                                          'voice-lstm1-3x128-0.978val0.991-AGDTfirst26last7.h5'))
    voice_dnn = tf.keras.models.load_model(os.path.join('models', 'voice-dnn-2x20-0.992val0.993-AGDTfirst26last7.h5'))
    gender_lstm1 = tf.keras.models.load_model(os.path.join('models',
                                                           'gender-lstm1-3x128-0.923val0.934-AGDTfirst26last7.h5'))
    gender_dnn = tf.keras.models.load_model(os.path.join('models', 'gender-dnn-2x20-0.952val0.937-AGDTfirst26last7.h5'))
    case_lstm1 = tf.keras.models.load_model(os.path.join('models',
                                                         'case-lstm1-3x128-0.934val0.962-AGDTfirst26last7.h5'))
    case_dnn = tf.keras.models.load_model(os.path.join('models', 'case-dnn-2x20-0.957val0.963-AGDTfirst26last7.h5'))
    degree_lstm1 = tf.keras.models.load_model(os.path.join('models',
                                                           'degree-lstm1-3x128-0.998val0.999-AGDTfirst26last7.h5'))
    degree_dnn = tf.keras.models.load_model(os.path.join('models', 'degree-dnn-2x20-0.999val0.999-AGDTfirst26last7.h5'))

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
    fpos = Morphs('pos', pos_tags, pos_lstm1, pos_dnn)
    fperson = Morphs('person', person_tags, person_lstm1, person_dnn)
    fnumber = Morphs('number', number_tags, number_lstm1, number_dnn)
    ftense = Morphs('tense', tense_tags, tense_lstm1, tense_dnn)
    fmood = Morphs('mood', mood_tags, mood_lstm1, mood_dnn)
    fvoice = Morphs('voice', voice_tags, voice_lstm1, voice_dnn)
    fgender = Morphs('gender', gender_tags, gender_lstm1, gender_dnn)
    fcase = Morphs('case', case_tags, case_lstm1, case_dnn)
    fdegree = Morphs('degree', degree_tags, degree_lstm1, degree_dnn)

    return fpos, fperson, fnumber, ftense, fmood, fvoice, fgender, fcase, fdegree


def elision_normalize(s):
    """Turn unicode characters which look similar to 2019 into 2019."""
    return s.replace("\u02BC", "\u2019").replace("\u1FBF", "\u2019").replace("\u0027", "\u2019").\
        replace("\u1FBD", "\u2019")


greek_text = 'νέος μὲν καὶ ἄπειρος δικῶν ἔγωγε ἔτι, δεινῶς δὲ καὶ ἀπόρως ἔχει μοι περὶ τοῦ πράγματος, ὦ ἄνδρες, ' \
             'τοῦτο μὲν εἰ ἐπισκήψαντος τοῦ πατρὸς ἐπεξελθεῖν τοῖς αὑτοῦ φονεῦσι μὴ ἐπέξειμι, τοῦτο δὲ εἰ ἐπεξιόντι ' \
             'ἀναγκαίως ἔχει οἷς ἥκιστα ἐχρῆν ἐν διαφορᾷ καταστῆναι, ἀδελφοῖς ὁμοπατρίοις καὶ μητρὶ ἀδελφῶν.'
annotator = 'Vanessa Gorman'

# Add LSTM2 to these when they are ready
pos, person, number, tense, mood, voice, gender, case, degree = create_morph_classes()
morphs = (pos, person, number, tense, mood, voice, gender, case, degree)

# Load character list and annotator list
with open(os.path.join('data', 'jsons', 'all_norm_characters.json'), encoding='utf-8') as json_file:
    all_norm_characters = json.load(json_file)
with open(os.path.join('data', 'jsons', 'annotators.json'), encoding='utf-8') as json_file:
    all_annotators = json.load(json_file)
with open(os.path.join('data', 'jsons', 'short_annotators.json'), encoding='utf-8') as json_file:
    short_annotators = json.load(json_file)

# Create the normalizer
normalise = Normaliser().normalise

# Create annotator tensor
annotator_tensor = [0] * 37
try:
    annotator_tensor[all_annotators.index(annotator)] = 1

# Make Vanessa Gorman the default annotator
except IndexError:
    annotator_tensor[0] = 1

blank_character_tensor = np.array([0]*174, dtype=np.bool_)

for word in greek_text.split():
    # The whole token tensor starts out blank because it's challenging to fill out the empty characters.
    token_tensor = np.array([blank_character_tensor]*21, dtype=np.bool_)

    # Normalize each token before tensorizing its characters.
    normalized_form = normalise(elision_normalize(word))[0]
    print(normalized_form)
    token_length = len(normalized_form)

    # # Create token tensors for tokens longer than 21 characters
    # if token_length > 21:
    #     token_tensor = []
    #     for character in normalized_form[:10]:
    #         character_tensor = [0]*137
    #         try:
    #             character_tensor[all_norm_characters.index(character)] = 1
    #         except ValueError:
    #             character_tensor[136] = 1
    #
    #         # Append the annotator tensor at the end of every character tensor
    #         character_tensor = character_tensor + annotator_tensor
    #         character_tensor = np.array(character_tensor, dtype=np.bool_)
    #         token_tensor.append(character_tensor)
    #     character_tensor = [0]*137
    #     character_tensor[135] = 1
    #
    #     # Append the annotator tensor at the end of every character tensor
    #     character_tensor = character_tensor + annotator_tensor
    #     character_tensor = np.array(character_tensor, dtype=np.bool_)
    #     token_tensor.append(character_tensor)
    #     for character in normalized_form[-10:]:
    #         character_tensor = [0]*137
    #         try:
    #             character_tensor[all_norm_characters.index(character)] = 1
    #         except ValueError:
    #             character_tensor[136] = 1
    #
    #         # Append the annotator tensor at the end of every character tensor
    #         character_tensor = character_tensor + annotator_tensor
    #         character_tensor = np.array(character_tensor, dtype=np.bool_)
    #         token_tensor.append(character_tensor)
    #     token_tensor = np.array(token_tensor, dtype=np.bool_)
    #
    # # Create token tensors for tokens shorter than 22 characters
    # else:
    #     for i, character in enumerate(normalized_form):
    #         character_tensor = [0]*137
    #         try:
    #             character_tensor[all_norm_characters.index(character)] = 1
    #         except ValueError:
    #             character_tensor[136] = 1
    #
    #         # Append the annotator tensor at the end of every character tensor
    #         character_tensor = character_tensor + annotator_tensor
    #         character_tensor = np.array(character_tensor)
    #         token_tensor[21-token_length+i] = character_tensor
    #
    # # Add each tensor token to the samples
    # py_samples.append(token_tensor)