from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import range
import os
import zipfile
import traceback

import spacy
from spacy_syllables import SpacySyllables
import pysubs2
import lap
from scipy.optimize import linear_sum_assignment, minimize
import numpy as np
import numpy.random as rand
import pandas as pd
import glob
from collections import defaultdict
from multiprocessing import Pool, TimeoutError, cpu_count

if __name__ == '__main__':
    from multiseq import MultiseqDataset, seq_collate
else:
    from .multiseq import MultiseqDataset, seq_collate

ENCODING = 'ISO-8859-1'

np.set_printoptions(suppress = True)

#TODO: Add download from URL maybe?
DOWNLOAD_URL = "https://app.box.com/s/604f3grzaudrq7z0qm3qyw65j2q5drr3/folder/111495067042"
TYPE_FACTOR = 7
IND_FACTOR = 1
SPACY_CODECS = {}
LANG2CODECS_MAP = {'en': 'en_core_web_md', 'es': 'es_core_news_md', 'fr': 'fr_core_news_md'}
DUMMY_WRD = '^'
TRAIN_SPLIT = 0.95
MAXLEN = 100
NUM_VIDS = 500

class SubtitlesDataset(MultiseqDataset):
    """Dataset of noisy spirals."""

    def __init__(self, modalities, base_dir, subdir='train',
                 truncate=False, item_as_dict=False):
        processed_dir = os.path.join(base_dir, 'processed', subdir)
        if not os.path.isdir(processed_dir):
            process_dataset(langs=modalities, data_dir=base_dir, subdir=subdir)

        regex = "{}_{}".format('_'.join(sorted(modalities)), '(\d+)_(\d+)\.csv')
        rates = 1.0
        base_rate = rates

        # Load x, y, and metadata as separate modalities
        preprocess = {
            # Keep only embedded value
            modalities[0]: lambda df : df.filter(regex=r'{}_encoding_.*'.format(modalities[0])),
            # Keep only embedded value
            modalities[1]: lambda df : df.filter(regex=r'{}_encoding_.*'.format(modalities[1]))
        }

        super(SubtitlesDataset, self).__init__(
            modalities, processed_dir, regex,
            [preprocess[m] for m in modalities],
            rates, base_rate, truncate, [], item_as_dict)

def filter_langs(modalities):
    for idx, m in enumerate(modalities):
        if not m in SPACY_CODECS:
            try:
                SPACY_CODECS[m] = spacy.load(LANG2CODECS_MAP[m])
                syllables = SpacySyllables(SPACY_CODECS[m])
                SPACY_CODECS[m].add_pipe(syllables, after='tagger')
            except Exception as e:
                print("Modality {} couldn't be decoded. Dropped.".format(m))
                del modalities[idx]

def process_dataset(langs=['en', 'es'], data_dir='./subtitles', subdir='train'):
    transcript_zip_path = os.path.join(data_dir, 'ted.zip')
    transcript_unzip_path = os.path.join(data_dir, 'ted')
    vidid_file = os.path.join(data_dir, "{}_{}_{}".format('vid', '_'.join(sorted(langs)), 'lang.txt'))
    processed_path = os.path.join(data_dir, 'processed', subdir)

    if not os.path.isdir(data_dir) \
        or not os.path.isfile(transcript_zip_path) \
        or not os.path.isfile(vidid_file):
        print("Please download and save as {} from {}.".format((transcript_zip_path, vidid_file), DOWNLOAD_URL))
        exit(1)

    if not os.path.isdir(transcript_unzip_path):
        with zipfile.ZipFile(transcript_zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    if not os.path.isdir(processed_path):
        os.makedirs(processed_path)

    with open(vidid_file) as f:
        lines = f.readlines()[:NUM_VIDS]

    if subdir == 'train':
        lines = lines[:int(len(lines) * TRAIN_SPLIT)]
    else:
        lines = lines[int(len(lines) * TRAIN_SPLIT):]

    pool = Pool(processes=cpu_count())
    res = []
    for file_count, line in enumerate(lines):
        vid_name = line.rstrip()
        dst_file =  os.path.join(processed_path, "{}_{:05d}_{}.csv".format('_'.join(sorted(langs)), file_count, '{:05d}'))
        src_files = {m:os.path.join(transcript_unzip_path, '{}_{}.srt'.format(vid_name, m)) for m in langs}
        tokens = defaultdict(list)
        res.append(pool.apply_async(process_srt, (src_files, dst_file, langs, tokens, subdir)))
        # process_srt(src_files, dst_file, langs, tokens, subdir)
    pool.close()
    pool.join()

def process_srt(src_files, dst_file, langs, tokens=defaultdict(list), subdir='train'):
    filter_langs(langs)
    try:
        subs = {}
        for lang in langs:
            subs[lang] = pysubs2.load(src_files[lang], encoding= ENCODING, format_= "srt")
            tokenize_sub(subs[lang], SPACY_CODECS[lang])
        if not len(subs['en']) == len(subs['es']):
            print("Mismatch in {}. Skipping.".format(",".join(src_files.values())))
            return
        align_subs(subs, tokens, langs)
    
        def _attributes(key, subs):
            for sen_idx, sub in enumerate(subs[key]):
                for token_idx, token in enumerate(sub.text):
                    if sub.shuffled:
                        word_idx = sub.order[token_idx]
                    else:
                        word_idx = token_idx
                    dummy_word = token.text == DUMMY_WRD
                    yield (token.text, token.pos_, sub.start, sub.end, sen_idx, word_idx, dummy_word, *token.vector.tolist())

        def _columns(key, subs):
            columns = list(range(7 + SPACY_CODECS[key].vocab.vectors.shape[1]))
            columns[0] = '{}_{}'.format(key, 'word')
            columns[1] = '{}_{}'.format(key, 'type')
            columns[2] = '{}_{}'.format(key, 'starttime')
            columns[3] = '{}_{}'.format(key, 'endtime')
            columns[4] = '{}_{}'.format(key, 'sentence_idx')
            columns[5] = '{}_{}'.format(key, 'word_idx')
            columns[6] = '{}_{}'.format(key, 'dummy_word')
            columns[7:] = ['{}_{}_{}'.format(key, 'encoding', ind) for ind in range(SPACY_CODECS[key].vocab.vectors.shape[1])]
            return columns

        for key in subs.keys():
            subs[key] = pd.DataFrame(_attributes(key, subs), columns=_columns(key, subs))
        df = pd.concat(subs.values(), axis=1)
        if subdir == 'train':
            for i in range(0, len(df), MAXLEN):
                df.loc[i:i+MAXLEN-1,:].to_csv(dst_file.format(int(i/MAXLEN)), index=False)
        else:
            df.to_csv(dst_file, index=False)
    except Exception as e:
        print("Couldn't process {}".format(src_files))
        traceback.print_exc()

def tokenize_sub(sub, codex):
    for idx, dialogue in enumerate(sub):
        if type(dialogue.text) == str:
            split = dialogue.text.split('\\N')
        else:
            split = dialogue.text.text.split('\\N')
        if len(split) > 1:
            dialogue.text = codex(split[0])
            starttime, endtime = split[3].split(' --> ')
            startime = pysubs2.make_time(s=pd.to_timedelta(starttime).seconds)
            endtime = pysubs2.make_time(s=pd.to_timedelta(endtime).seconds)
            sub.insert(idx, pysubs2.SSAEvent(start=startime, end=endtime, text=codex(split[4])))
        else:
            if type(dialogue.text) == str:
                dialogue.text = codex(dialogue.text)
            else:
                dialogue.text = codex(dialogue.text.text)

def align_subs(subs, tokens, langs):
    keys = subs.keys()
    for sub in zip(*(subs[k] for k in keys)):
        assert(len(sub) == 2), "Only 2 modalities supported currently"
        X, Y = sub[0].text, sub[1].text
        cost_matrix = get_cost_mat(X, Y)
        _, x_order, y_order = lap.lapjv(cost_matrix)
        new_X_words = [x.text for x in X]
        new_Y_words = [y.text for y in Y]
        if len(new_X_words) < len(new_Y_words):
            new_X_words.extend([DUMMY_WRD] * (len(Y) - len(X)))
        else:
            new_Y_words.extend([DUMMY_WRD] * (len(X) - len(Y)))
        
        new_X = SPACY_CODECS[X.lang_](' '.join(new_X_words))
        new_Y = SPACY_CODECS[Y.lang_](' '.join([new_Y_words[an] for an in x_order]))
        for idy, an in enumerate(x_order):
            if an < len(Y):
                new_Y[idy].pos_ = Y[an].pos_
            else:
                new_Y[idy].pos_ = 'PUNCT'
        for idx, an in enumerate(y_order):
            if idx < len(X):
                new_X[idx].pos_ = X[idx].pos_       # Not shuffled
            else:
                new_X[idx].pos_ = 'PUNCT'

        sub[0].shuffled, sub[0].order, sub[0].text = False, x_order, new_X
        sub[1].shuffled, sub[1].order, sub[1].text = True, y_order, new_Y

def get_cost_mat(X, Y):
    size = max(len(X), len(Y))
    cost_mat = np.ones((size, size))*100
    
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            type_cost = (x.pos_ != y.pos_)
            ind_cost = abs(i-j)        
            cost_mat[i, j] = type_cost * TYPE_FACTOR + ind_cost * IND_FACTOR
    return cost_mat

def test_dataset(modalities = ['en', 'es'], data_dir='./subtitles'):
    print("Loading data...")
    dataset = SubtitlesDataset(modalities, data_dir)
    print(len(dataset))

    print("Testing batch collation...")
    data = seq_collate([dataset[i] for i in range(min(10, len(dataset)))])
    
    print("Batch shapes:")
    for d in data[:-2]:
        print(d.shape)
    
    print("Sequence lengths: ", data[-1])
    
    print("Checking through data for mismatched sequence lengths...")
    for i, data in enumerate(dataset):
    
        print("Sequence: ", dataset.seq_ids[i])
        x, y = data[:2]

        print(x.shape, y.shape)
        
        if len(x) != len(y):
            print("WARNING: Mismatched sequence lengths.")

if __name__ == '__main__':
    test_dataset()

