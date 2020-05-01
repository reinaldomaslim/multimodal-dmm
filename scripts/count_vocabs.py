import glob
import pandas as pd 
import numpy as np 
import os 
import spacy 
from spacy.tokens import Doc
from spacy.vocab import Vocab
from random import shuffle

ENCODING = 'ISO-8859-1'

def unicode2str(word):
    if isinstance(word, str):
        return word
    elif isinstance(word, float):
        return str(word)
    else:
        return word.decode(ENCODING)


modalities = ['en', 'es', 'fr']
vocabs = {}
docs = {}
for modal in modalities:
    vocabs[modal] = set()
    docs[modal] = Doc(Vocab())

csvs = glob.glob('../datasets/subtitles/*.csv')
shuffle(csvs)

for csv in csvs:
    print(csv)
    languages = csv.split('/')[-1].split('.')[0].split('_')[:2]
    df = pd.read_csv(csv)
    
    first_lang = df.loc[:, ['0', '2']].to_numpy()
    second_lang = df.loc[:, ['4', '6']].to_numpy()

    for i in range(first_lang.shape[0]):
        word, hash_val = first_lang[i]
        word = unicode2str(word)
        docs[languages[0]].vocab.strings.add(word)
        vocabs[languages[0]].add(hash_val)
        
        word, hash_val = second_lang[i]
        word = unicode2str(word)
        docs[languages[1]].vocab.strings.add(word)
        vocabs[languages[1]].add(hash_val)    

for modal in modalities:
    print('-------------')
    print(modal)
    print(len(vocabs[modal]))

    for i in range(10):
        word = docs[modal].vocab.strings[vocabs[modal].pop()]
        print('test doc:', word)