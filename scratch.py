import os
from bs4 import BeautifulSoup
from collections import Counter
import pickle
import numpy as np
from utilities_morph import elision_normalize
from greek_normalisation.normalise import Normaliser, Norm

folder = os.path.join('data', 'pickles')
for file in os.listdir(folder):
    if file[0] == 'l':
        with open(os.path.join(folder, file), 'rb') as infile:
            old_file = pickle.load(infile)
        if type(old_file) is not np.ndarray:
            print(f'{file} not numpy. Converting to array...')
            new_file = np.array(old_file, dtype=np.bool_)
            with open(os.path.join(folder, file), 'wb') as outfile:
                pickle.dump(new_file, outfile)
            print(f'{file} converted to numpy array!')
        else:
            print(f'{file} is already in good shape.')
