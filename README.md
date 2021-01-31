# AnGEL Morphology
## An Ancient Greek Morphology Tagger
Angel takes in Ancient Greek plain text and returns morphology tags for each token. More specifically, it tags part-of-speech, person, number, tense, mood, voice, gender, case, and degree. The morphology tagging mostly follows the [AGDT 2.0 style](https://github.com/PerseusDL/treebank_data/blob/master/AGDT2/guidelines/Greek_guidelines.md#mph_tgs). The exception would be the inclusion of the "g" tag for the part-of-speech "particle" which is not included in the AGDT 2.0 documentation, but is used by some annotators within the [AGDT 2.1 treebank](https://github.com/PerseusDL/treebank_data/tree/master/v2.1/Greek) collection. For a list of all tags available within each element, refer to the [AGDT 2.1 tagset](https://github.com/PerseusDL/treebank_data/blob/master/v2.1/Greek/TAGSETS.xml)

## Installation
    pip install angel-tag

## Usage
The input should be a string. The output is a tuple of tuples.

    from angel import tag

    greek_string = 'ὧν ἐς πολὺ μὲν οὐκ ἐπῄσθοντο Ῥωμαῖοι διὰ τὰς ἐν ἄστει κρίσεις τε καὶ στάσεις·'
    results = tag(greek_string)
    print(results)
Output:

    (('ὧν', 'p-p---ng-'), ('ἐς', 'r--------'), ('πολὺ', 'a-s---na-'), ('μὲν', 'd--------'), ('οὐκ', 'd--------'), ('ἐπῄσθοντο', 'v3paim---'), ('Ῥωμαῖοι', 'n-p---mn-'), ('διὰ', 'r--------'), ('τὰς', 'l-p---fa-'), ('ἐν', 'r--------'), ('ἄστει', 'n-s---nd-'), ('κρίσεις', 'n-p---fa-'), ('τε', 'd--------'), ('καὶ', 'c--------'), ('στάσεις', 'n-p---fa-'), ('·', 'u--------'))

If you just need to tag one sentence, then the above example is fine. But if you want to tag an entire document, don't 
feed it to the tagger one 
sentence at a time; or even worse, one token at a time. It'll take forever that way and accuracy will suffer. Just
give it the entire document all at once.

    from angel import tag

    with open('xenophon-hellenica.txt', 'r', encoding='utf-8') as infile:
        entire_book = infile.read()

    results = tag(entire_book)

It is possible that an excessively large string may cause memory issues. If you run into that problem, then
perhaps split the text in half and try that. This is an issue that will be addressed in later releases.

## Design
This novel architecture utilizes no rules or morphology lookup tables. Rather, it examines individual token morphology and each token's context within the sentence using a series of neural networks. Furthermore, because of the varying tendencies of the many human annotators which are found among the AGDT treebanks, Angel considered annotators as a feature during training. Consequently, while running inference, an annotator must be chosen for the tagger to emulate. "Vanessa Gorman" is the default choice as her annotation style is up to date and she is currently the single most prolific annotator. 

## Accuracy
Partially imitating the assessment criteria used by [Barbara McGillivray and Alessandro Vatri](https://www.researchgate.net/publication/328791830_The_Diorisis_Ancient_Greek_Corpus) in the development of their state of the art (91% POS accuracy) tagger they used in their Diorisis corpus, Angel was trained on 26 works in the [AGDT 2.1 treebank](https://github.com/PerseusDL/treebank_data/tree/master/v2.1/Greek) while 7 works were reserved for validation during training. Though Diorisis trained on roughly 50% more data (from the [PROIEL treebanks](https://github.com/proiel/proiel-treebank/)), Angel outperformed it scoring 95.5% accuracy predicting parts-of-speech in the validation set. That score was further confirmed by testing upon the first five works within the [Gorman treebanks](https://github.com/perseids-publications/gorman-trees) wherein it scored 95.7% part-of-speech accuracy, earning it state of the art status by a significant margin.

## Further Development
* Exceptionally large works may cause memory issues. This will be addressed in subsequent updates.
* Annotation inconsistencies will be addressed. Some obsolete tags still exist in the current version of AGDT.
* Lemmatization and dependency tagging integration is planned. Other paradigms for morphology tagging may be 
  implemented as well.
* A lot more training data is available. It will be used.
* Much hyperparameter tinkering to be done.
* OS testing.

## License
Copyright (c) 2021 Chris Drymon under the [MIT license](https://github.com/chrisdrymon/greek-morph-tagger/blob/master/LICENSE).
