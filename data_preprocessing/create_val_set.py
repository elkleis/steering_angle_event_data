"""
Prints the training files that should be used as validation files during the training process

Usage:
$ ./create_val_set.py --source_folder <output_dir> --percentage_split <percentage>
"""

import os
import numpy as np
import argparse
import glob
import h5py
from itertools import groupby
from operator import itemgetter


def split_sequence(data):
    sequences = []
    for k, g in groupby(enumerate(data), lambda (i, x): i - x):
        sequences.append(map(itemgetter(1), g))
    return sequences


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DDD17
    parser.add_argument('--source_folder', default='/mnt/Seagate/DDD17/ddd17/model/data',
                        help='Path training set.')
    # DDD20
    # parser.add_argument('--source_folder', default='/mnt/Seagate/DDD17/ddd20/model/data',
    #                     help='Path training set.')
    parser.add_argument('--percentage_split', default='0.1', type=float,
                        help='Percentage of the training set to be used as the validation set')
    args = parser.parse_args()

    recordings = glob.glob(args.source_folder + '/*.hdf5')

    # Find the number of images that belong to the training set
    tot_len = 0
    for rec in recordings:
        f_in = h5py.File(rec, 'r')
        train_idxs = f_in['train_idxs'][()]
        tot_len += len(train_idxs)
        f_in.close()

    # Calculate the images that will be used as validation set
    val_num_idxs = int(np.floor(tot_len * args.percentage_split))
    val_idxs = []

    for rec in sorted(recordings, reverse=True):
        f_in = h5py.File(rec, 'r+')
        train_idxs = f_in['train_idxs'][()]
        filename = os.path.basename(rec)

        if val_num_idxs:
            # If the sequence is smaller or equal to the validation, all the whole recording will be used
            if len(train_idxs) < val_num_idxs:
                print("File: '{}'. Images used: {} (whole file included)".format(filename, len(train_idxs)))
                val_num_idxs -= len(train_idxs)
            # If the sequence is larger than the validations set, then the rest of validation images will be included
            elif len(train_idxs) == val_num_idxs:
                print("File: '{}'. Images used: {} (whole file included)".format(filename, len(train_idxs)))
                val_num_idxs -= len(train_idxs)
            else:
                print("File: '{}'. Images used: {}".format(filename, val_num_idxs))
                val_num_idxs = 0
        else:
            break

        f_in.close()

    print('\nDone. Validation and train split calculated.')
