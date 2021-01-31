from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Religion',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: Greek',
    'Natural Language :: English',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.7',
    'Topic :: Text Processing :: Linguistic'
]

setup(
    name='angel-tag',
    version='0.0.3',
    description='An Ancient Greek Morphology Tagger',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='',
    author='Chris Drymon',
    author_email='chrisdrymon@yahoo.com',
    license='MIT',
    classifiers=classifiers,
    keywords=['greek', 'ancient greek', 'morphology', 'classics', 'computational linguistics'],
    packages=find_packages(),
    install_requires=['tensorflow',
                      'numpy',
                      'gdown',
                      'greek_normalisation',
                      'gensim']
)