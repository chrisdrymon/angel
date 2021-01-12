import os
import tensorflow as tf
from bs4 import BeautifulSoup
import numpy as np
import json


# Create a custom model saver
class ModelSaver(tf.keras.callbacks.Callback):
    """A custom tensorflow model saver that returns useful information"""
    def __init__(self, morph_title, nn_type):
        super().__init__()
        self.best_val_acc = 0
        self.best_epoch = 0
        self.new_best = False
        self.morph_title = morph_title
        self.nn_type = nn_type

    def on_train_begin(self, logs=None):
        self.best_val_acc = 0
        self.new_best = False

    def on_epoch_end(self, epoch, logs=None):
        # Save the best model based on validation accuracy.
        if logs['val_accuracy'] > self.best_val_acc:
            self.best_val_acc = logs['val_accuracy']
            model_name = os.path.join('models', f'{self.morph_title}-{self.nn_type}-1x20-{logs["accuracy"]:.3f}'
                                                f'val{logs["val_accuracy"]:.3f}')
            tf.keras.models.save_model(self.model, model_name, save_format='h5')
            self.best_epoch = epoch + 1
            self.new_best = True
            print('\nModel saved at epoch', epoch + 1, 'with', self.best_val_acc, 'validation accuracy.\n')

    def on_train_end(self, logs=None):
        if self.new_best:
            print('\nBest Model saved at epoch', self.best_epoch, 'with', self.best_val_acc, 'validation accuracy.')


class Morphs:
    """Hold data for one aspect of morphology."""
    def __init__(self, title, tags, lstm):
        self.title = title
        self.tags = tags
        self.lstm = lstm


def create_morph_classes():
    # Load each trained model for testing
    pos_lstm = tf.keras.models.load_model(os.path.join('models', 'pos-1x64-0.945val0.907'))
    pos_dnn = tf.keras.models.load_model(os.path.join('models', 'pos-DNN-1x20-0.925val0.914'))
    person_lstm = tf.keras.models.load_model(os.path.join('models', 'person-1x64-0.995val0.979'))
    number_lstm = tf.keras.models.load_model(os.path.join('models', 'number-1x64-0.990val0.967'))
    tense_lstm = tf.keras.models.load_model(os.path.join('models', 'tense-1x64-0.996val0.973'))
    mood_lstm = tf.keras.models.load_model(os.path.join('models', 'mood-1x64-0.995val0.978'))
    voice_lstm = tf.keras.models.load_model(os.path.join('models', 'voice-1x64-0.996val0.977'))
    gender_lstm = tf.keras.models.load_model(os.path.join('models', 'gender-1x64-0.962val0.909'))
    case_lstm = tf.keras.models.load_model(os.path.join('models', 'case-1x64-0.977val0.934'))
    degree_lstm = tf.keras.models.load_model(os.path.join('models', 'degree-1x64-0.999val0.999'))

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
    pos = Morphs('pos', pos_tags, pos_lstm)
    person = Morphs('person', person_tags, person_lstm)
    number = Morphs('number', number_tags, number_lstm)
    tense = Morphs('tense', tense_tags, tense_lstm)
    mood = Morphs('mood', mood_tags, mood_lstm)
    voice = Morphs('voice', voice_tags, voice_lstm)
    gender = Morphs('gender', gender_tags, gender_lstm)
    case = Morphs('case', case_tags, case_lstm)
    degree = Morphs('degree', degree_tags, degree_lstm)

    return pos, person, number, tense, mood, voice, gender, case, degree


def create_samples_and_labels(morphs, corpora_size):
    """Return samples and labels. Focused on individual morphs, not training everything."""
    corpora_folder = os.path.join('corpora', 'greek', 'annotated', 'perseus-771dca2', 'texts')
    with open(os.path.join('jsons', 'all_characters.json'), encoding='utf-8') as json_file:
        all_characters = json.load(json_file)
    if isinstance(corpora_size, int):
        indir = os.listdir(corpora_folder)[:corpora_size]
    else:
        indir = os.listdir(corpora_folder)
    file_count = 0
    # Search through every work in the annotated Greek folder
    for file in indir:
        if file[-4:] == '.xml':
            file_count += 1
            print(file_count, file)
            for morph in morphs:
                morph.correct_prediction_count = 0
                morph.labels = []

            # A list to hold the token tensors
            samples = []
            token_count = 0

            # Open the files (they are XML's) with beautiful soup and search through every word in every sentence.
            xml_file = open(os.path.join(corpora_folder, file), 'r', encoding='utf-8')
            soup = BeautifulSoup(xml_file, 'xml')
            sentences = soup.find_all('sentence')
            for sentence in sentences:
                tokens = sentence.find_all(['word', 'token'])
                for token in tokens:
                    if token.has_attr('form') and token.has_attr('postag'):

                        # In the AGDT corpus, 218 unique characters occur. Hence the size of the character tensor.
                        blank_character_tensor = np.array([0] * 219, dtype=np.bool_)

                        # The longest word in the AGDT corpus is 21 characters long
                        wordform_tensor = np.array([blank_character_tensor] * 21, dtype=np.bool_)
                        wordform_length = len(token['form'])

                        # For each character in the token, create a one-hot tensor
                        for i, character in enumerate(token['form']):
                            character_tensor = np.array([0] * 219, dtype=np.bool_)
                            try:
                                character_tensor[all_characters.index(character)] = 1
                            except ValueError:
                                character_tensor[218] = 1
                            wordform_tensor[21 - wordform_length + i] = character_tensor

                        # This tensor collects all the wordform tensors
                        samples.append(wordform_tensor)

                        # Creates a labels tensor to check predictions against
                        for i, morph in enumerate(morphs):
                            try:
                                morph.labels.append(token['postag'][i])
                            except IndexError:
                                morph.labels.append('-')
    return morphs
