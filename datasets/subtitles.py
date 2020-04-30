from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import range
import os

import numpy as np
import numpy.random as rand
import pandas as pd
import glob

if __name__ == '__main__':
    from multiseq import MultiseqDataset, seq_collate
else:
    from .multiseq import MultiseqDataset, seq_collate

class SubtitlesDataset(MultiseqDataset):
    """Dataset of noisy spirals."""

    def __init__(self, modalities, base_dir):

        regex = modalities[0]+'_'+modalities[1]+'_(\d+)\.csv'
        rates = 1.0
        base_rate = rates

        # Load x, y, and metadata as separate modalities
        preprocess = {
            # Keep only embedded value
            modalities[0]: lambda df : df.loc[:, ['2']],
            # Keep only embedded value
            modalities[1]: lambda df : df.loc[:, ['6']]
        }

        super(SubtitlesDataset, self).__init__(
            modalities, base_dir, regex,
            [preprocess[m] for m in modalities],
            rates, base_rate, False, [], False)

def test_dataset(modalities = ['en', 'es'], data_dir='./subtitles'):
    print("Loading data...")
    dataset = SubtitlesDataset(modalities, data_dir)
    
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

