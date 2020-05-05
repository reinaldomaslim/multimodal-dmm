import os
import spacy
from spacy_syllables import SpacySyllables
import pysubs2
import numpy as np
import numpy.random as rand
import pandas as pd

ENCODING = 'ISO-8859-1'
SPACY_CODECS = {}
LANG2CODECS_MAP = {'en': 'en_core_web_md', 'es': 'es_core_news_md', 'fr': 'fr_core_news_md'}
DUMMY_WRD = '^'

def csv2srt(fcsv,  data_dir='../datasets/subtitles'):
    print("loading data...")
    df = pd.read_csv(fcsv)
    fname = fcsv.split('/')[-1].split('.')[0].split('_')[-1]
    result_path = data_dir+'/result'
    print(result_path)
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    #extract and corrupted words as test
    vectors = {}
    for key in df.keys():
        if 'encoding' in key:
            modal = key.split('_')[0]
            try: vectors[modal] 
            except: vectors[modal] = []     
            vectors[modal].append(df[key].to_numpy(dtype = np.float32))

    words, orderings, times = {}, {}, {}
    for modal in vectors.keys():
        SPACY_CODECS[modal] = spacy.load(LANG2CODECS_MAP[modal])
        words[modal] = df[modal+'_word'].to_numpy()
        orderings[modal] = np.stack((df[modal+'_sentence_idx'].to_numpy(), df[modal+'_word_idx'].to_numpy())).T
        times[modal] = np.stack((df[modal+'_starttime'].to_numpy(), df[modal+'_endtime'].to_numpy())).T
        vectors[modal] = np.asarray(vectors[modal]).T

        # corrupt_ind = np.random.binomial(1, 0.9, size = words[modal].shape[0]).astype(np.bool)
        # words[modal][corrupt_ind] = DUMMY_WRD
    

    #get back corrupted words from vectors
    print("vector to words...")
    for modal in vectors.keys():
        for i in range(vectors[modal].shape[0]):
            if words[modal][i] == DUMMY_WRD and np.sum(vectors[modal][i])>0:
                key = SPACY_CODECS[modal].vocab.vectors.most_similar(vectors[modal][i][np.newaxis, :])[0][0][0]
                word = SPACY_CODECS[modal].vocab.strings[key].lower()
                words[modal] = word

    print("reordering words")
    #reorder words
    ordered_words = {}
    for modal in vectors.keys():
        sentence_idxs = np.unique(orderings[modal][:, 0])
        ordered_words[modal] = []
        for sentence_idx in sentence_idxs:
            ind = orderings[modal][:, 0] == sentence_idx
            if np.sum(ind) == 0: continue
            word_idx = orderings[modal][:, 1][ind].astype(np.int32)
            res = words[modal][ind][word_idx]
            dummy_ind = res != DUMMY_WRD
            ordered_words[modal].append([res[dummy_ind].tolist(), times[modal][:, 0][ind][0], times[modal][:, 1][ind][1]])

    print("saving srt...")
    #create srt
    separator = ' '
    for modal in vectors.keys():
        subs = pysubs2.SSAFile()
        for dialogue in ordered_words[modal]:
            sub = pysubs2.SSAEvent(start = dialogue[1], end = dialogue[2], text = separator.join(dialogue[0]))
            subs.append(sub)
        subs.save(result_path + '/res_'+fname+'_'+modal+'.srt')

if __name__ == '__main__':
    fcsv = '../datasets/subtitles/processed/train/en_es_00000.csv'
    csv2srt(fcsv)