import numpy as np
import os
import pandas as pd

AVAILABLE_SPLITS = ['train', 'test', 'test_scaffolds']

BASE_PATH = os.path.dirname(__file__)


# BASE_PATH = os.path.join(BASE_PATH, "..")


def get_dataset(split='train'):
    """
    Loads MOSES dataset

    Arguments:
        split (str): split to load. Must be
            one of: 'train', 'test', 'test_scaffolds'

    Returns:
        list with SMILES strings
    """
    if split not in AVAILABLE_SPLITS:
        raise ValueError(
            f"Unknown split {split}. "
            f"Available splits: {AVAILABLE_SPLITS}"
        )
    if split not in AVAILABLE_SPLITS:
        raise ValueError(
            f"Unknown split {split}. "
            f"Available splits: {AVAILABLE_SPLITS}")
    path = os.path.join(BASE_PATH, 'data', split + '.csv.gz')
    smiles = pd.read_csv(path, compression='gzip')['SMILES'].values
    return smiles


def get_statistics(split='test'):
    path = os.path.join(BASE_PATH, 'data', split + '_stats.npz')
    return np.load(path, allow_pickle=True)['stats'].item()
