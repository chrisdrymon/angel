import os
from bs4 import BeautifulSoup
import time
import json
import datetime
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from collections import Counter
# from tensor_utils import header, poser, person, grammatical_number, tenser, mooder, voicer, gender, caser, lemmer

# Enable this to run on CPU instead of GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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
            model_name = os.path.join('models',
                                      f'pos-1x128-{logs["accuracy"]:.3f}val{logs["val_accuracy"]:.3f}-fullAGDT')
            tf.keras.models.save_model(model, model_name, save_format='h5')
            # The following is the save command that doesn't work.
            # tf.keras.models.save_model(model, model_name)
            self.best_epoch = epoch + 1
            self.new_best = True
            print('\n\nModel saved at epoch', epoch + 1, 'with', self.best_val_acc, 'validation accuracy.\n')

    def on_train_end(self, logs=None):
        if self.new_best:
            print('\nBest Model saved at epoch', self.best_epoch, 'with', self.best_val_acc, 'validation accuracy.')


corpora_folder = os.path.join('corpora', 'greek', 'annotated', 'perseus-771dca2', 'texts')
indir = os.listdir(corpora_folder)
file_count = 0
py_samples = []
py_labels = []

with open(os.path.join('jsons', 'all_characters.json'), encoding='utf-8') as json_file:
    all_characters = json.load(json_file)
pos_tags = ['l', 'n', 'a', 'r', 'c', 'i', 'p', 'v', 'd', 'm', 'g', 'u']
person_tags = ['1', '2', '3']
number_tags = ['s', 'p', 'd']
tense_tags = ['p', 'i', 'r', 'l', 't', 'f', 'a']
mood_tags = ['i', 's', 'n', 'm', 'p', 'o']
voice_tags = ['a', 'p', 'm', 'e']
gender_tags = ['m', 'f', 'n']
case_tags = ['n', 'g', 'd', 'a', 'v']
degree_tags = ['p', 'c', 's']

# Change this for each aspect of morphology
relevant_tagset = pos_tags

# Search through every work in the annotated Greek folder
for file in indir:
    if file[-4:] == '.xml':
        file_count += 1
        print(file_count, file)
        # Open the files (they are XML's) with beautiful soup and search through every word in every sentence.
        xml_file = open(os.path.join(corpora_folder, file), 'r', encoding='utf-8')
        soup = BeautifulSoup(xml_file, 'xml')
        sentences = soup.find_all('sentence')
        for sentence in sentences:
            tokens = sentence.find_all(['word', 'token'])
            for token in tokens:
                # Enable this if the search should ignore elliptical tokens
                # if token.has_attr('artificial') is False and token.has_attr('empty-token-sort') is False:
                # The longest token is 21 characters. 218 unique characters occur in the corpus.
                if token.has_attr('form') and token.has_attr('postag'):
                    blank_character_tensor = np.array([0]*219, dtype=np.bool_)
                    token_tensor = np.array([blank_character_tensor]*21, dtype=np.bool_)
                    token_length = len(token['form'])
                    for i, character in enumerate(token['form']):
                        character_tensor = np.array([0]*219, dtype=np.bool_)
                        try:
                            character_tensor[all_characters.index(character)] = 1
                        except ValueError:
                            character_tensor[218] = 1
                        token_tensor[21-token_length+i] = character_tensor
                    py_samples.append(token_tensor)

                    # Now create the label tensors.
                    # Go change save file's name right now.
                    # For each aspect of morphology, refactor this tensor's name.
                    pos_tensor = [0] * (len(relevant_tagset) + 1)
                    try:
                        # For each aspect of morphology, change the postag position.
                        pos_tensor[relevant_tagset.index(token['postag'][0])] = 1
                    except IndexError:
                        print(sentence['id'], token['id'], token['form'])
                        pos_tensor[-1] = 1
                    except ValueError:
                        print(sentence['id'], token['id'], token['form'])
                        pos_tensor[-1] = 1
                    py_labels.append(pos_tensor)
samples = np.array(py_samples, dtype=np.bool_)
labels = np.array(py_labels, dtype=np.bool_)
print(f'Samples: {len(samples)}')
print(f'Labels: {len(labels)}')

print('Converting to Numpy Arrays. This may take a few minutes...')
# Split data into an 80%/20% training/validation split.
split = int(.8*len(labels))
train_data = np.array(samples[:split], dtype=np.bool_)
val_data = np.array(samples[split:], dtype=np.bool_)
train_labels = np.array(labels[:split], dtype=np.bool_)
val_labels = np.array(labels[split:], dtype=np.bool_)

# This setting keeps Tensorflow from automatically reserving all my GPU's memory
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

# Enter the samples and labels into Tensorflow to train a neural network
model = tf.keras.Sequential()
model.add(layers.Bidirectional(layers.LSTM(128, activation='relu'), input_shape=(21, 219)))
model.add(layers.Dense(len(relevant_tagset) + 1, activation='softmax'))
modelSaver = ModelSaver()

# log_dir = "data\\tf_logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
# # log_dir = os.path.join('data', 'nn_models', datetime.datetime.now().strftime("%Y%m%d-%H%M"))
# tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)  # tensorboard --logdir data/tf_logs
#
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=40, validation_data=(val_data, val_labels), verbose=2,
          callbacks=[modelSaver])
