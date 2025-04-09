# Small Language Models

Tis project aims to implement the Virterbi algorithm in part-of-speech (POS) tagging, as well as language modelling using Hidden Markov Models (HMMs), and Bigram models.

## Setup

Setup a python virtual environment and install dependencies in requirements.txt

```bash
$ python -m venv venv

$ source venv/bin/activate

$ pip install -r requirements.txt
```

Enter the list of languages in the `main.py`

### Downloading datasets

Datasets can be downloaded from the universal dependencies framework [#](universaldependencies.org/#download)

The datasets used for development and testing are:

- [UD English Atis](https://universaldependencies.org/treebanks/en_atis/index.html)
- [UD Old East Slavic TOROT](https://universaldependencies.org/treebanks/orv_torot/index.html)
- [UD Turkish Atis](https://universaldependencies.org/treebanks/tr_atis/index.html)

## Running

```bash
$ python main.py
```

