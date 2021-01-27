# AnGEL Morphology
## An Ancient Greek Morphology Tagger
Angel takes in Ancient Greek plain text and returns morphology tags for each token. More specifically, it tags part-of-speech, person, number, tense, mood, voice, gender, case, and degree. The morphology tagging mostly follows the [AGDT 2.0 style](https://github.com/PerseusDL/treebank_data/blob/master/AGDT2/guidelines/Greek_guidelines.md#mph_tgs). The exception would be the inclusion of the "g" tag for the part-of-speech "particle" which is not included in the AGDT 2.0 documentation, but is used by some annotators within the [AGDT 2.1 treebank](https://github.com/PerseusDL/treebank_data/tree/master/v2.1/Greek) collection. For a list of all tags available within each element, refer to the [AGDT 2.1 tagset](https://github.com/PerseusDL/treebank_data/blob/master/v2.1/Greek/TAGSETS.xml)

## Design
This novel architecture utilizes no rules or morphology lookup tables. Rather, it examines individual token morphology and each token's context within the sentence using a series of neural networks. Furthermore, because of the varying tendencies of the many human annotators which are found among the AGDT treebanks, Angel considered annotators as a feature during training. Consequently, while running inference, an annotator must be chosen for the tagger to emulate. "Vanessa Gorman" is the default choice as her annotation style is up to date and she is currently the single most prolific annotator. 

The initial model loading time as well as the manner in which they were trained encourage the processing of an entire document at a time rather than feeding the tagger individual sentences or tokens at a time.

## Accuracy
Partially imitating the assessment criteria used by [Barbara McGillivray and Alessandro Vatri](https://www.researchgate.net/publication/328791830_The_Diorisis_Ancient_Greek_Corpus) in the development of their state of the art (91% POS accuracy) tagger they used in their Diorisis corpus, AnGEL was trained on 26 works in the [AGDT 2.1 treebank](https://github.com/PerseusDL/treebank_data/tree/master/v2.1/Greek) while 7 works were reserved for validation during training. Though Diorisis trained on roughly 50% more data (from the [PROIEL treebanks](https://github.com/proiel/proiel-treebank/)), Angel outperformed it scoring 95.5% accuracy in predicting parts-of-speech. That score was further confirmed by testing upon the first five works within the [Gorman treebanks](https://github.com/perseids-publications/gorman-trees) wherein it scored 95.7% part-of-speech accuracy, earning it state of the art status by a significant margin.

## Usage
This repo in its current state is intended to "show my work." If you just want to use the end-product, 11_AnGEL.py is what you want. The hard work has been done. Just clone the repo, get the appropriate libraries, and modify the "greek_text" variable in 11_AnGEL.py to suit your input needs. For something even more user friendly, a PIP package will be released shortly.

## Further Development
* Exceptionally large works may cause memory issues. This will be addressed in subsequent updates.
* Annotation inconsistencies will be addressed. Some obsolete tags still exist in the current version of AGDT.
* Much hyperparameter tinkering to be done.
* A PIP release is coming
* A lot more training data is available. It will be used.
* Lemmatization and dependency tagging integration is planned. Other paradigms for morphology tagging may be implemented as well.
