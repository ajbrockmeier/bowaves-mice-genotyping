#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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


def main(df_filename, test_fold, split, train_valid_test, strain=None, tsc=None, source=None, hours_in_segment=None, n_segments=None,
         window_length_s=2, n_windows=40000, centroid_length_s=1, n_centroids=200,
         use_sign_invariant=False,use_spectral=False,use_sphere=False,hpc=False):

    strains = ('BXD87', 'DBA2', 'C57B6')
    tscs = ('Het', 'WT')
    sources = ('Jax_Lab')

    if source is None:
        source = 'Jax_Lab'

    if n_segments is None:
        n_segments = 480 # how many possibly overlapping randomly drawn segments (will be evenly divided across mice of given type)

    if hours_in_segment is None:
        hours_in_segment = 1   # hours_in_segment = 1

    mice_df = pd.read_pickle('meta-data/'+df_filename)
    if hpc:
        print('In HPC')
        mice_df['filepaths'] = mice_df['filepaths'].apply(strip_leading_from_list_filenames)
    print(mice_df['filepaths'])
    nfolds, nsplits = get_split_info(mice_df)
    print(nsplits, nfolds)

    n_minutes_segment = 60 * hours_in_segment
    sfreq = mice_df['sfreq'].iloc[0] # assume all mice are the same, should be checked on any new dataset
    segment_length = int(sfreq * 60 * n_minutes_segment)  # in samples
    window_length = int(sfreq * window_length_s)  # in samples

    if use_spectral:
        param_string = 'spectral' + '_'+ str(n_windows) + '_' + str(window_length) + '_' + str(n_centroids) + '_' + str(centroid_length_s)
    else:
        centroid_len = int(sfreq * centroid_length_s)  # in samples
        if use_sign_invariant:
            str_si = '_si'
        else:
            str_si = '_sv'
        if use_sphere:
            str_sph = '_sph'
        else:
            str_sph = ''
        param_string = str(n_windows) + '_' + str(window_length) + '_' + str(n_centroids) + '_' + str(centroid_len) + str_si +str_sph


    waveform_dict = load_centroid_dict(test_fold, split, param_string)
    df = mice_df[mice_df['source'] == source]

    assert source in sources, 'Jax_Lab is only choices'
    assert train_valid_test in ['train', 'valid', 'test'], 'train/test/valid are only choices'
    assert split in range(nsplits), 'split parameter is invalid'
    assert test_fold in range(nfolds), 'test fold is invalid'

    if tsc is None or strain is None: # run all
        segment_info, all_counts, all_labels = get_counts_by_type_fold(df, test_fold, split, train_valid_test, n_segments, segment_length, window_length, waveform_dict, use_sign_invariant, use_spectral)
        pickle.dump(segment_info, open( "data/segment_info_"+train_valid_test+"_"+str(n_segments)+"_"+str(split)+str(test_fold)+param_string+".pickle", "wb"))
        pickle.dump(all_counts, open( "data/count_"+train_valid_test+"_"+str(n_segments)+str(split)+str(test_fold)+param_string+".pickle", "wb"))
        pickle.dump(all_labels, open( "data/cluster_"+train_valid_test+"_"+str(n_segments)+str(split)+str(test_fold)+param_string+".pickle", "wb"))
    else:
        print(strain, tsc)
        assert strain in strains, 'Need to specify mouse line'
        assert tsc in tscs, 'Need to specify mouse line'
        df = mice_df[(mice_df['source'] == source) & (mice_df['tsc'] == tsc) & (mice_df['strain'] == strain)]
        segment_indices, name_list, counts, labels = get_counts(df, test_fold, split, train_valid_test, n_segments, segment_length, window_length, waveform_dict, use_sign_invariant, use_spectral)
        segment_info = (segment_indices,name_list)
        pickle.dump(segment_info, open( "data/segment_info_"+train_valid_test+"_"+str(n_segments)+"_"+str(split)+str(test_fold)+param_string+strain+tsc+".pickle", "wb" ) )
        pickle.dump(counts, open( "data/count_"+train_valid_test+"_"+str(n_segments)+str(split)+str(test_fold)+param_string+strain+tsc+".pickle", "wb" ) )
        pickle.dump(labels, open( "data/cluster_"+train_valid_test+"_"+str(n_segments)+str(split)+str(test_fold)+param_string+strain+tsc+".pickle", "wb" ) )

def load_centroid_dict(test_fold,split,param_string):
    strains = ('BXD87', 'DBA2', 'C57B6')  # tsc
    tscs = ('Het', 'WT')  # genotype
    print('Starting consolidate dictionary', test_fold, split)

    dict_filename = "models/centroids_" + str(split) + str(test_fold) + '_' + param_string + ".pickle"
#    dict_filename = "models/centroids_all_" + param_string + ".pickle" # fully consolidated
    try:
        waveform_dict = pickle.load(open(dict_filename, "rb"))
        print('Consolidated dictionary ready!')
    except:
        print('Running dictionary consolidation')
        waveform_dict = dict()
        n_success = 0
        for strain in strains:
            for tsc in tscs:
                key_dict= (split, test_fold, strain, tsc)
                try:
                    centroid_type = str(split) + str(test_fold) + strain + tsc + param_string
                    centroids = pickle.load(open("data/centroids_" + centroid_type + ".pickle", "rb"))
                    waveform_dict[key_dict] = centroids
                    n_success += 1
                except:
                    print(tsc, strain, split, test_fold, ' is not finished')

        print(n_success, len(strains) * len(tscs))
        if n_success == len(strains) * len(tscs):
            print('Writing data')
            pickle.dump(waveform_dict, open(dict_filename, "wb"))
        else:
            print('Not all types have centroids ')
            raise

    return waveform_dict


if __name__ == '__main__':
    strains = {0:'BXD87', 1:'DBA2', 2:'C57B6'}
    tscs = {0:'Het', 1:'WT'}
    sources = ('Jax_Lab')

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sources,default='Jax_Lab',
                        help="dataset source")
    parser.add_argument("--learn", choices=['train','valid', 'test'], default='train',
                        help="which split for classifier")
    parser.add_argument("--split", type=int, default=0,
                        help="which random split for dictionary training")
    parser.add_argument("--fold", type=int, default=0,
                        help="which fold in the split for dictionary training")
    parser.add_argument("--class1", type=int, choices=[0,1,2],
                        help="index of class subtype-1")
    parser.add_argument("--class2", type=int, choices=[0,1],
                        help="index of class subtype-2")
    parser.add_argument("--windows", type=int, default=40000,
                        help="how many windows were used in training")
    parser.add_argument("--window_length_s", type=float, default=2,
                        help="how many seconds each window")
    parser.add_argument("-k", "--n_centroids", type=int, default=200,
                        help="how many centroids")
    parser.add_argument("--centroid_length_s", type=float, default=1,
                        help="how many seconds is each waveform/centroid")
    parser.add_argument("--segments", type=int, default=480,
                        help="how many segments to get from each class")
    parser.add_argument("--hours_in_segment", type=float,
                        help="how many hours in each segment for the count vectors")
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
    if args.class1 is not None:
        args.class1 = strains[args.class1]
    if args.class2 is not None:
        args.class2 = tscs[args.class2]
    main(args.df, args.fold, args.split, args.learn, args.class1, args.class2, args.source, args.hours_in_segment, args.segments,
         args.window_length_s, args.windows, args.centroid_length_s, args.n_centroids,args.use_sign_invariant,args.use_spectral,
         args.use_sphere,
         args.hpc)

