"""Training code for spirals dataset."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import range
import os, copy

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection

from datasets.subtitles import *
from datasets.multiseq import seq_collate_dict
import models
import trainer
import yaml
import pandas as pd

class SubtitlesTrainer(trainer.Trainer):
    """Class for training on subtitle datasets."""

    parser = copy.copy(trainer.Trainer.parser)

    # Add these arguments specifically for the Spirals dataset
    parser.add_argument('--rectify_srts', type=yaml.safe_load,
                        default={}, metavar='DICT',
                        help='rectifying data subdirectory')
    parser.add_argument('--dst_file', type=str, 
                        default='./subtitles_save/rectified_{}.srt',
                        help='device to use')

    # Set parameter defaults for spirals dataset
    defaults = {
        'modalities' : ['en', 'es'],
        'batch_size' : 3, 'split' : 1, 'bylen' : False,
        'epochs' : 100, 'lr' : 1e-4,
        'kld_anneal' : 100, 'burst_frac' : 0.1,
        'drop_frac' : 0.1, 'start_frac' : 0.25, 'stop_frac' : 0.75,
        'eval_metric' : 'mse', 'viz_metric' : 'mse',
        'eval_freq' : 10, 'save_freq' : 10,
        'data_dir' : './datasets/subtitles',
        'save_dir' : './subtitles_save'
    }
    parser.set_defaults(**defaults)

    def build_model(self, constructor, args):
        """Construct model using provided constructor."""
        dims = {'en': 300, 'es': 50}
        dists = {'en': 'Normal',
                 'es': 'Normal',}
        z_dim = args.model_args.get('z_dim', 64)
        h_dim = args.model_args.get('h_dim', 64)
        n_layers = args.model_args.get('n_layers', 3)
        gauss_out = (args.model != 'MultiDKS')        
        encoders = {'en': models.common.DeepGaussianMLP(dims['en'], z_dim, h_dim, n_layers),
                    'es': models.common.DeepGaussianMLP(dims['es'], z_dim, h_dim, n_layers)}
        decoders = {'en': models.common.DeepGaussianMLP(z_dim, dims['en'], h_dim, n_layers),
                    'es': models.common.DeepGaussianMLP(z_dim, dims['es'], h_dim, n_layers)}
        custom_mods = [m for m in ['en', 'es'] if m in args.modalities]
        model = constructor(args.modalities,
                            dims=(dims[m] for m in args.modalities),
                            dists=[dists[m] for m in args.modalities],
                            encoders={m: encoders[m] for m in custom_mods},
                            decoders={m: decoders[m] for m in custom_mods},
                            z_dim=z_dim, h_dim=h_dim,
                            device=args.device, **args.model_args)
        return model

    def pre_build_args(self, args):
        """Process args before model is constructed."""
        args = super(SubtitlesTrainer, self).pre_build_args(args)
        # Set up method specific model and training args
        if args.method in ['b-skip', 'f-skip', 'b-mask', 'f-mask']:
            # No direct connection from features to z in encoder
            args.model_args['feat_to_z'] = True
            # Do not add unimodal ELBO training loss for RNN methods
            args.train_args['uni_loss'] = True
        return args

    def post_build_args(self, args):
        """Process args after model is constructed."""
        # Default reconstruction loss multipliers
        if args.rec_mults == 'auto':
            dims = self.model.dims
            corrupt_mult = 1 / (1 - args.corrupt.get('uniform', 0.0))
            args.rec_mults = {m : ((1.0 / dims[m]) / len(args.modalities)
                                   * corrupt_mult)
                              for m in args.modalities}
        return args

    def load_data(self, modalities, args):
        """Loads data for specified modalities."""
        print("Loading data...")
        data_dir = os.path.abspath(args.data_dir)
        train_data = SubtitlesDataset(modalities, data_dir, mode='train',
                                      truncate=True, item_as_dict=True)
        test_data = SubtitlesDataset(modalities, data_dir, mode='test',
                                     truncate=True, item_as_dict=True)
        print("Done.")
        if len(args.normalize) > 0:
            print("Normalizing ", args.normalize, "...")
            # Normalize test data using training data as reference
            test_data.normalize_(modalities=args.normalize,
                                 ref_data=train_data)
            # Normalize training data in-place
            train_data.normalize_(modalities=args.normalize)
        return train_data, test_data

    def compute_metrics(self, model, infer, prior, recon,
                        targets, mask, lengths, order, args):
        """Compute evaluation metrics from batch of inputs and outputs."""
        metrics = dict()
        if type(lengths) != torch.Tensor:
            lengths = torch.FloatTensor(lengths).to(args.device)
        # Compute and store KLD and reconstruction losses
        metrics['kld_loss'] = model.kld_loss(infer, prior, mask).item()
        metrics['rec_loss'] = model.rec_loss(targets, recon, mask,
                                             args.rec_mults).item()

        for m in list(recon.keys()): targets[m][torch.isnan(targets[m])] = 0
        # Compute mean squared error in 2D space for each time-step
        tdims = targets[m].dim()
        mse = sum([(recon[m][0]-targets[m]).pow(2).sum(dim=list(range(2, tdims))) for m in list(recon.keys())])
        # Average across timesteps, for each sequence
        def time_avg(val):
            val[1 - mask.squeeze(-1)] = 0.0
            return val.sum(dim = 0) / lengths
        metrics['mse'] = time_avg(mse)[order].tolist()
        return metrics
 
    def summarize_metrics(self, metrics, n_timesteps):
        """Summarize and print metrics across dataset."""
        summary = dict()
        for key, val in list(metrics.items()):
            if type(val) is list:
                # Compute mean and std dev. of metric over sequences
                summary[key] = np.mean(val)
                summary[key + '_std'] = np.std(val)
            else:
                # Average over all timesteps
                summary[key] = val / n_timesteps
        print(('Evaluation\tKLD: {:7.1f}\tRecon: {:7.1f}\t' +
               'MSE: {:6.3f} +-{:2.3f}')\
              .format(summary['kld_loss'], summary['rec_loss'],
                      summary['mse'], summary['mse_std']))
        return summary

    def visualize(self, results, metric, args):
        return

    def save_results(self, results, args):
        pass

    def rectify_srt(self, args):
        rectify_dataset = SubtitlesDataset(args.modalities, args.data_dir,
                                           mode='rectify', truncate=True,
                                           item_as_dict=True, rectify_files=args.rectify_srts)
        assert isinstance(rectify_dataset.df, pd.DataFrame), "DataFrame not loaded."
        data, mask, lengths, order, ids = seq_collate_dict([rectify_dataset[0]])
        infer, prior, recon = self.model(data, lengths=lengths)

        def _update_row(row):
            new_row = pd.Series()
            for modal in args.modalities:
                for regex_part in ['type', 'starttime', 'endtime', 'sentence_idx', 'word_idx', 'dummy_word']:
                    key = '{}_{}'.format(modal, regex_part)
                    new_row[key] = row[key]

                vector_len = row.filter(regex=r'{}_encoding_.*'.format(modal)).shape[0]
                vector_idx = ['{}_encoding_{}'.format(modal, i) for i in range(vector_len)]
                if row['{}_word'.format(modal)] == PLCHLDR_WRD:
                    vector = recon[modal][0][row.name].detach().numpy()  # Asumming dataframe index is same as data index
                    key = SPACY_CODECS[modal].vocab.vectors.most_similar(vector)[0][0][0]
                    new_row['{}_word'.format(modal)] = SPACY_CODECS[modal].vocab.strings[key].lower()
                    new_row = new_row.append(pd.Series(vector[0], vector_idx))
                else:
                    new_row['{}_word'.format(modal)] = row['{}_word'.format(modal)]
                    new_row = new_row.append(row[vector_idx])
            return new_row

        rectified_df = rectify_dataset.df.apply(_update_row, axis=1)
        process_csv(rectified_df, args.dst_file)

if __name__ == "__main__":
    args = SubtitlesTrainer.parser.parse_args()
    trainer = SubtitlesTrainer(args)
    if args.rectify_srts:
        trainer.rectify_srt(args)
    else:
        trainer.run(args)