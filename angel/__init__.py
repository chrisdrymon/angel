import os
from greek_normalisation.normalise import Normaliser
import numpy as np
from gensim.models import KeyedVectors
from tensorflow.keras.models import load_model
import tensorflow as tf
import tarfile
import gdown


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
    pos_lstm1 = load_model(os.path.join(model_folder, 'pos-lstm1-3x128-0.927val0.939-AGDTfirst26last7.h5'))
    pos_dnn = load_model(os.path.join(model_folder, 'pos-dnn-2x20-0.939val0.942-AGDTfirst26last7.h5'))
    pos_lstm2 = load_model(os.path.join(model_folder, 'pos-lstm2-3x128-0.958val0.955-AGDTfirst26last7.h5'))

    print('Person models loading...')
    person_lstm1 = load_model(os.path.join(model_folder, 'person-lstm1-3x128-0.983val0.990-AGDTfirst26last7.h5'))
    person_dnn = load_model(os.path.join(model_folder, 'person-dnn-2x20-0.994val0.992-AGDTfirst26last7.h5'))
    person_lstm2 = load_model(os.path.join(model_folder, 'person-lstm2-3x128-0.994val0.994-AGDTfirst26last7.h5'))

    print('Number models loading...')
    number_lstm1 = load_model(os.path.join(model_folder, 'number-lstm1-3x128-0.955val0.980-AGDTfirst26last7.h5'))
    number_dnn = load_model(os.path.join(model_folder, 'number-dnn-2x20-0.977val0.981-AGDTfirst26last7.h5'))
    number_lstm2 = load_model(os.path.join(model_folder, 'number-lstm2-3x128-0.985val0.987-AGDTfirst26last7.h5'))

    print('Tense models loading...')
    tense_lstm1 = load_model(os.path.join(model_folder, 'tense-lstm1-3x128-0.976val0.990-AGDTfirst26last7.h5'))
    tense_dnn = load_model(os.path.join(model_folder, 'tense-dnn-2x20-0.990val0.992-AGDTfirst26last7.h5'))
    tense_lstm2 = load_model(os.path.join(model_folder, 'tense-lstm2-3x128-0.986val0.992-AGDTfirst26last7.h5'))

    print('Mood models loading...')
    mood_lstm1 = load_model(os.path.join(model_folder, 'mood-lstm1-3x128-0.981val0.992-AGDTfirst26last7.h5'))
    mood_dnn = load_model(os.path.join(model_folder, 'mood-dnn-2x20-0.994val0.992-AGDTfirst26last7.h5'))
    mood_lstm2 = load_model(os.path.join(model_folder, 'mood-lstm2-3x128-0.994val0.995-AGDTfirst26last7.h5'))

    print('Voice models loading...')
    voice_lstm1 = load_model(os.path.join(model_folder, 'voice-lstm1-3x128-0.978val0.991-AGDTfirst26last7.h5'))
    voice_dnn = load_model(os.path.join(model_folder, 'voice-dnn-2x20-0.992val0.993-AGDTfirst26last7.h5'))
    voice_lstm2 = load_model(os.path.join(model_folder, 'voice-lstm2-3x128-0.989val0.993-AGDTfirst26last7.h5'))

    print('Gender models loading...')
    gender_lstm1 = load_model(os.path.join(model_folder, 'gender-lstm1-3x128-0.923val0.934-AGDTfirst26last7.h5'))
    gender_dnn = load_model(os.path.join(model_folder, 'gender-dnn-2x20-0.952val0.937-AGDTfirst26last7.h5'))
    gender_lstm2 = load_model(os.path.join(model_folder, 'gender-lstm2-4x128-0.960val0.958-AGDTfirst26last7.h5'))

    print('Case models loading...')
    case_lstm1 = load_model(os.path.join(model_folder, 'case-lstm1-3x128-0.934val0.962-AGDTfirst26last7.h5'))
    case_dnn = load_model(os.path.join(model_folder, 'case-dnn-2x20-0.957val0.963-AGDTfirst26last7.h5'))
    case_lstm2 = load_model(os.path.join(model_folder, 'case-lstm2-2x128-0.975val0.977-AGDTfirst26last7.h5'))

    print('Degree models loading...')
    degree_lstm1 = load_model(os.path.join(model_folder, 'degree-lstm1-3x128-0.998val0.999-AGDTfirst26last7.h5'))
    degree_dnn = load_model(os.path.join(model_folder, 'degree-dnn-2x20-0.999val0.999-AGDTfirst26last7.h5'))
    degree_lstm2 = load_model(os.path.join(model_folder, 'degree-lstm2-2x128-0.998val0.999-AGDTfirst26last7.h5'))

    # The possible tags for each aspect of morphology
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
    """Place spaces around punctuation so it can be easily split into its own token later."""
    return fsentence.replace(',', ' , ').replace('·', ' · ').replace(';', ' ; ').replace('.', ' . ').\
        replace('?', ' ? ').replace('»', ' » ').replace('«', ' « ').replace('“', ' “ ').replace('„', ' „ ').\
        replace('(', ' ( ').replace(')', ' ) ').replace('>', ' > ').replace('<', ' < ').replace(':', ' : ').\
        replace('‘', ' ‘ ')


def vector_lookup(gword):
    """Return a vector for a given Greek word."""
    try:
        return wv[gword]
    except KeyError:
        return np.array([0]*100)


def tag(greek_text, annotator='Vanessa Gorman'):
    """Take in a string of Greek text and return that text morphologically tagged."""
    print('Loading models...')
    pos, person, number, tense, mood, voice, gender, case, degree = create_morph_classes()
    morphs = (pos, person, number, tense, mood, voice, gender, case, degree)

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
    return_list = []

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
    print("Angel's looking at each word by itself...")
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
    print('Reconsidering tags...')
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

    # In the future, convert to a tf.data format:
    # dataset = tf.data.Dataset.from_tensor_slices(padded_lstm2_input).window(15, 1, 1)

    # Run outputs through LSTM2
    print("Studying each word in light of its context...")
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
        return_list.append((token, pos.predicted_tags3[i] + person.predicted_tags3[i] + number.predicted_tags3[i] +
                            tense.predicted_tags3[i] + mood.predicted_tags3[i] + voice.predicted_tags3[i] +
                            gender.predicted_tags3[i] + case.predicted_tags3[i] + degree.predicted_tags3[i]))
    return tuple(return_list)


# This will keep Tensorflow quieter.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load some tuples that'll be needed
all_norm_characters = ("ζ", "ε", "ύ", "ς", "μ", "έ", "ν", "ἀ", "φ", "ί", "κ", "τ", "ω", "ρ", "ἐ", "π", "δ", "ο",
                       "ι", "ό", "σ", "λ", "ἡ", "ά", "θ", "̓", "ψ", "α", "υ", ".", "ῦ", "χ", "γ", "ᾳ", ",", "ὔ",
                       "ἵ", "η", "ή", "ῳ", "ῖ", "ὐ", "ξ", "ἰ", "β", "ῆ", "ῶ", "ἅ", "ἄ", "ὅ", "ὖ", "ώ", "ᾶ", "ἱ",
                       ";", "ὦ", "ὕ", "ὁ", "·", "ἑ", "ὑ", "ὄ", "ἔ", "ῇ", "ὀ", "ἁ", "ὧ", "ἴ", "-", "ῤ", "ἶ", "ὶ",
                       "[", "]", "ΐ", "ἠ", "ὡ", "ὤ", "ἕ", "ἥ", "ῥ", "ᾷ", "ἆ", "ῷ", "ὠ", "ῃ", "ἤ", "ῄ", "ἦ", "ἧ",
                       "ᾔ", "?", " ", "\"", "ᾠ", "ἷ", "ὥ", "ᾖ", "ᾤ", "ῴ", "ὗ", "ϊ", "ᾇ", "ᾧ", "(", ")", "ᾁ", "ᾗ",
                       "ᾴ", "ᾡ", "ᾐ", "ᾑ", "ΰ", "ᾀ", "ᾕ", "ᾆ", "†", "¯", "̆", "ᾄ", ">", "ϋ", "ῗ", "ᾦ", "<", "2",
                       "0", "’", ":", "—", "（", "）", "ᾅ", "ῧ", "ϝ")
all_annotators = ("Vanessa Gorman", "david.bamman", "david", "millermo2", "gleason", "Sean Stewart",
                  "Robert Gorman", "Francesco Mambrini", "Daniel Lim Libatique", "Alex Lessie", "James C. D'Amico",
                  "Brian Livingston", "Calliopi Dourou", "C. Dan Earley", "Connor Hayden", "Francis Hartel",
                  "George Matthews", "J. F. Gentile", "Jennifer Adams", "Jessica Nord", "Jennifer Curtin",
                  "Mary Ebbott", "Meg Luthin", "Molly Miller", "Michael Kinney", "Jack Mitchell", "Sam Zukoff",
                  "Scott J. Dube", "Tovah Keynton", "W. B. Dolan", "Florin Leonte", "Anthony D. Yates",
                  "Jordan Hawkesworth", "Giuseppe G. A. Celano", "Yoana Ivanova", "Polina Yordanova")

# This should place the models in a predictable place no matter the OS.
model_folder = os.path.join(os.path.expanduser('~'), 'angel_models')

# See if models have been downloaded. If not, download them.
if os.path.isdir(model_folder) and 'fasttext.wordvectors' in os.listdir(model_folder):
    wv = KeyedVectors.load(os.path.join(model_folder, 'fasttext.wordvectors'))
else:
    # Download the models if they don't exist
    print('Ancient Greek language models need to be downloaded. Once downloaded, the models will be saved locally. '
          'This should only be required once. It may take a minute. These are big files.')
    print('Downloading models...')

    # URL needs to look like https://drive.google.com/uc?id=1-w6Ld2r1Z9kKI2oZ3RWTsmNdJSb12hxP
    url = 'https://drive.google.com/uc?id=1MPTNoRNnTEY818BOdCjRhs7nSG3PgRWW'
    output = 'models.tar.xz'
    gdown.download(url, output, quiet=False)
    extract_folder = os.path.join(os.path.expanduser('~'), 'angel_models')

    print(f'Unpacking models to {extract_folder}...')
    tar = tarfile.open(output, 'r:xz')
    tar.extractall(path=extract_folder)
    tar.close()

    # Delete the downloaded file
    os.remove(output)

    # Load the word vectors
    wv = KeyedVectors.load(os.path.join(extract_folder, 'fasttext.wordvectors'))
