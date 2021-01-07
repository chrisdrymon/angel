import os
from bs4 import BeautifulSoup
import time
import json
import datetime
import tensorflow as tf
from tensorflow.keras import layers
from collections import Counter
# from tensor_utils import header, poser, person, grammatical_number, tenser, mooder, voicer, gender, caser, lemmer

# Enable this to run on CPU instead of GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# This setting keeps Tensorflow from automatically reserving all my GPU's memory
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


# Create a custom model saver
class ModelSaver(tf.keras.callbacks.Callback):
    # A custom tensorflow model saver that returns useful information
    best_loss = 100
    best_acc = 0
    best_val_acc = 0
    best_epoch = 0
    best_lr = 0
    best_dropout = 0
    new_best = False

    def on_train_begin(self, logs=None):
        self.best_loss = 100
        self.best_acc = 0
        self.best_val_acc = 0
        self.new_best = False

    def on_epoch_end(self, epoch, logs=None):
        # Save the best model based on validation accuracy.
        if logs['val_accuracy'] > self.best_val_acc:
            self.best_val_acc = logs['val_accuracy']
            # os.path.join('data', 'tf_logs', "m{0:.3f}val{1:.3f}".format(logs['accuracy'], logs['val_accuracy']))
            model_name = os.path.join('data', 'models', "m{0:.3f}val{1:.3f}".format(logs['accuracy'],
                                                                                    logs['val_accuracy']))
            tf.keras.models.save_model(model, model_name, save_format='h5')
            # The following is the save command that doesn't work.
            # tf.keras.models.save_model(model, model_name)
            self.best_epoch = epoch + 1
            self.new_best = True
            print('\n\nModel saved at epoch', epoch + 1, 'with', self.best_val_acc, 'validation accuracy.\n')

    def on_train_end(self, logs=None):
        if self.new_best:
            print('\nBest Model saved at epoch', self.best_epoch, 'with', self.best_val_acc, 'validation accuracy.')


pos0_dict = {'a': 'adj', 'n': 'noun', 'v': 'verb', 'd': 'adv', 'c': 'conj', 'g': 'conj', 'r': 'adposition', 'b': 'conj',
             'p': 'pronoun', 'l': 'article', 'i': 'interjection', 'x': 'other', 'm': 'numeral', 'e': 'interjection'}
pos1_dict = {'1': 'first', '2': 'second', '3': 'third'}
pos2_dict = {'s': 'singular', 'p': 'plural', 'd': 'dual'}
pos3_dict = {'p': 'present', 'i': 'imperfect', 'r': 'perfect', 'a': 'aorist', 'l': 'pluperfect', 'f': 'future', 't':
             'future perfect'}
pos4_dict = {'i': 'indicative', 's': 'subjunctive', 'n': 'infinitive', 'm': 'imperative', 'p': 'participle',
             'o': 'optative'}
pos5_dict = {'a': 'active', 'm': 'middle', 'p': 'passive', 'e': 'middle or passive'}
pos6_dict = {'m': 'masculine', 'f': 'feminine', 'n': 'neuter'}
pos7_dict = {'n': 'nominative', 'g': 'genitive', 'd': 'dative', 'v': 'vocative', 'a': 'accusative'}
proiel_pos_dict = {'A': 'adj', 'D': 'adv', 'S': 'article', 'M': 'numeral', 'N': 'noun', 'C': 'conj', 'G': 'conj',
                   'P': 'pronoun', 'I': 'interjection', 'R': 'adposition', 'V': 'verb'}

# The purpose is to extract data from the annotated corpora to be used to train a machine learning algorithm. Two goals
# are in focus. 1) Be able to correctly identify the POS of any occurrence of the lemma ο. 2) Be able to identify the
# head of the lemma ο if it is acting as an article.

# 1.1) Consider all tokens which occur 4 words before the article to 10 words after the article [-4, 10]. Out of the
# 79,335 articles which have non-elliptical heads in this corpus, only 73 have heads which which occur outside of that
# window.
# 2.1) If the head is elliptical, recognize that.

corpora_folder = os.path.join('corpora', 'greek', 'annotated', 'perseus-771dca2', 'texts')
indir = os.listdir(corpora_folder)
file_count = 0
samples = []
labels = []

with open(os.path.join('jsons', 'all_characters.json'), encoding='utf-8') as json_file:
    all_characters = json.load(json_file)
pos_tags = ['l', 'n', 'a', 'r', 'c', 'i', 'p', 'v', 'd', 'm', 'g', 'u']

# Search through every work in the annotated Greek folder
for file in indir:
    if file[-4:] == '.xml' and file[:6] == 'tlg000':
        file_count += 1
        print(file_count, file)
        # Open the files (they are XML's) with beautiful soup and search through every word in every sentence.
        xml_file = open(os.path.join(corpora_folder, file), 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'xml')
        sentences = soup.find_all('sentence')
        for sentence in sentences:
            tokens = sentence.find_all(['word', 'token'])
            for token in tokens:
                # Enable this is the search should ignore elliptical tokens
                # if token.has_attr('artificial') is False and token.has_attr('empty-token-sort') is False:
                # The longest token is 21 characters. 218 unique characters occur in the corpus.
                if token.has_attr('form') and token.has_attr('postag'):
                    blank_character_tensor = [0]*218
                    # It's like this for testing. Change to [character_tensor]*21 later.
                    token_tensor = [blank_character_tensor]*21
                    token_length = len(token['form'])
                    for i, character in enumerate(token['form']):
                        character_tensor = [0]*218
                        character_tensor[all_characters.index(character)] = 1
                        token_tensor[21-token_length+i] = character_tensor
                    samples.append(token_tensor)

                    # Now create the label tensors
                    pos_tensor = [0]*13
                    try:
                        pos_tensor[pos_tags.index(token['postag'][0])] = 1
                    except IndexError:
                        print(sentence['id'], token['id'], token['form'])
                        pos_tensor[12] = 1
                    except ValueError:
                        print(sentence['id'], token['id'], token['form'])
                        pos_tensor[12] = 1
                    labels.append(pos_tensor)
print(f'Samples: {len(samples)}')
print(f'Labels: {len(labels)}')

# Split data into an 80%/20% training/validation split.
split = int(.8*len(labels))
train_data = samples[:split]
val_data = samples[split:]
train_labels = labels[:split]
val_labels = labels[split:]

# Enter the samples and labels into Tensorflow to train a neural network
model = tf.keras.Sequential()
# Input_shape = (n_steps, n_features). Since I combined everything into one tensor, try 49 features for now.
model.add(layers.Bidirectional(layers.LSTM(50, activation='relu'), input_shape=(21, 218)))
# model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, activation='relu', dropout=.3)))
# model.add(layers.Bidirectional(layers.LSTM(128, activation='relu')))
model.add(layers.Dense(13, activation='softmax'))
modelSaver = ModelSaver()
#
# log_dir = "data\\tf_logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
# # log_dir = os.path.join('data', 'nn_models', datetime.datetime.now().strftime("%Y%m%d-%H%M"))
# tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)  # tensorboard --logdir data/tf_logs
#
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=5, validation_data=(val_data, val_labels), verbose=2)
