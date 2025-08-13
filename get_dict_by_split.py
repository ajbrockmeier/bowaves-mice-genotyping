#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from mice_utils import *
import pickle
import argparse


def strip_leading_from_list_filenames(list_filenames):
    l2 = []
    for f in list_filenames:
        if f[0] == '/':
            l2.append(f[1:])
        else:
            l2.append(f)
    return l2


def main(df_filename, test_fold, split, strain, tsc, window_length_s, n_windows, centroid_length_s, n_centroids,
         use_sign_invariant=False, use_sphere=False, use_spectral=False, hpc=False):
    strains = ('BXD87', 'DBA2', 'C57B6')
    tscs = ('Het', 'WT')
    source = 'Jax_Lab'

    mice_df = pd.read_pickle('meta-data/'+df_filename)
    if hpc:
        print('In HPC')
        mice_df['filepaths'] = mice_df['filepaths'].apply(strip_leading_from_list_filenames)
    print(mice_df['filepaths'])
    nfolds, nsplits = get_split_info(mice_df)
    print(nsplits, nfolds)

    sfreq = mice_df['sfreq'].iloc[0] # assume all mice are the same, should be checked on any new dataset
    centroid_len = int(sfreq * centroid_length_s)  # in samples
    window_length = int(sfreq * window_length_s)  # in samples

    if strain is not None and tsc is not None:
        print(strain,tsc)
        assert strain in strains, 'Need to specify mouse line'
        assert tsc in tscs, 'Need to specify mouse line'
    else:
        print('Running all is not implemented')
        raise

    assert split in range(nsplits), 'split parameter is invalid'
    assert test_fold in range(nfolds), 'test fold is invalid'
    if use_spectral:
        nfft = centroid_length_s
        param_string = 'spectral' + '_'+ str(n_windows) + '_' + str(window_length) + '_' + str(n_centroids) + '_' + str(nfft)
    else:
        if use_sign_invariant:
            str_si = '_si'
        else:
            str_si = '_sv'
        if use_sphere:
            str_sph = '_sph'
        else:
            str_sph = ''
        param_string = str(n_windows) + '_' + str(window_length) + '_' + str(n_centroids) + '_' + str(centroid_len) + str_si +str_sph

    centroids_param_string = str(split)+str(test_fold) + strain + tsc + param_string
    print('Running:', centroids_param_string)
    time_start = time.time()
    df = mice_df[(mice_df['source'] == source) & (mice_df['tsc'] == tsc) & (mice_df['strain'] == strain)]
    windows, window_indices, mice_names = grab_windows_train(df, test_fold, split, n_windows, window_length)
    time_end = time.time()
    a_time = time_end - time_start
    print("time in minutes (windows):", a_time / 60)

    if use_spectral:
        centroids, labels, inertia, iters = create_spectral_codebooks(windows, n_centroids, nfft=nfft)
        train_info = (mice_names, window_indices, labels, inertia, iters)
    else:
        centroids, labels, shifts, inertia, iters = create_codebooks(windows, n_centroids, centroid_len, use_sign_invariant)
        train_info = (mice_names, window_indices, labels, shifts, inertia, iters)

    time_end = time.time()
    a_time = time_end - time_start
    print("time in minutes (centroids):", a_time / 60)

    pickle.dump(centroids, open("data/centroids_" + centroids_param_string + ".pickle", "wb"))
    pickle.dump(train_info, open("data/centroids_info_"+centroids_param_string+".pickle", "wb"))


if __name__ == '__main__':
    strains = {0:'BXD87', 1:'DBA2', 2:'C57B6'}
    tscs = {0:'Het', 1:'WT'}
    sources = ('Jax_Lab')

    parser = argparse.ArgumentParser()
#    parser.add_argument("--source", choices=sources, default='Jax_Lab',
#                        help="dataset source")
    parser.add_argument("--split", type=int, default=0,
                        help="which random split for dictionary training")
    parser.add_argument("--fold", type=int, default=0,
                        help="which fold in the split for dictionary training")
    parser.add_argument("--class1", type=int, choices=[0,1,2],
                        help="index of class subtype-1")
    parser.add_argument("--class2", type=int, choices=[0,1],
                        help="index of class subtype-2")
    parser.add_argument("--windows", type=int, default=40000,
                        help="how many windows to use in training")
    parser.add_argument("--window_length_s", type=float, default=2,
                        help="how many seconds each window")
    parser.add_argument("-k", "--n_centroids", type=int, default=200,
                        help="how many centroids")
    parser.add_argument("--centroid_length_s", type=float, default=1,
                        help="how many seconds is each waveform/centroid")
    parser.add_argument("--df", type=str, default='Jax_mice_with_splits_df.pickle',
                        help="filename in the meta/data path")
    parser.add_argument('--sign_invariant', dest='use_sign_invariant', action='store_true')
    parser.set_defaults(use_sign_invariant=False)
    parser.add_argument('--sphere', dest='use_sphere', action='store_true')
    parser.set_defaults(use_sphere=False)
    parser.add_argument('--spectral', dest='use_spectral', action='store_true')
    parser.set_defaults(use_spectral=False)
    parser.add_argument('--hpc', action='store_true')
    parser.add_argument('--local', dest='hpc', action='store_false')
    parser.set_defaults(hpc=True)
    args = parser.parse_args()

    if args.class1 is not None: # if None run all
        args.class1 = strains[args.class1]
    else:
        print('running all classes is not implemented')

    if args.class2 is not None: # if None run all
        args.class2 = tscs[args.class2]
    else:
        print('running all classes is not implemented')

    main(args.df, args.fold, args.split, args.class1, args.class2,
         args.window_length_s, args.windows, args.centroid_length_s, args.n_centroids, args.use_sign_invariant,args.use_sphere,
         args.use_spectral, args.hpc)

