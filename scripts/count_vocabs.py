import glob
import pandas as pd 
import numpy as np 
import os 
import spacy 


modalities = ['en', 'es', 'fr']
vocabs = {}
for modal in modalities:
    vocabs[modal] = set()

csvs = glob.glob('../datasets/subtitles/*.csv')

for csv in csvs:
    print(csv)
    languages = csv.split('/')[-1].split('.')[0].split('_')[:2]
    df = pd.read_csv(csv)
    
    first_lang_hashes = df.loc[:, ['2']].to_numpy().squeeze()
    second_lang_hashes = df.loc[:, ['6']].to_numpy().squeeze()

    for i in range(first_lang_hashes.shape[0]):
        hash_val = first_lang_hashes[i]
        vocabs[languages[0]].add(hash_val)


    for i in range(second_lang_hashes.shape[0]):
        hash_val = second_lang_hashes[i]
        vocabs[languages[1]].add(hash_val)

for modal in modalities:
    print('-------------')
    print(modal)
    print(len(vocabs[modal]))