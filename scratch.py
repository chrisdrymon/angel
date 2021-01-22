import os
from bs4 import BeautifulSoup
from collections import Counter
import pickle
import numpy as np
from utilities_morph import elision_normalize
from greek_normalisation.normalise import Normaliser, Norm

folder = os.path.join('data', 'npys')
for file in os.listdir(folder):
    with open(os.path.join(folder, file), 'rb') as infile:
        old_file = np.load(infile)
    with open(os.path.join(folder, f'{file[:-3]}pickle'), 'wb') as outfile:
        pickle.dump(old_file, outfile)
    print(f'Wrote {file[:-3]}pickle')
