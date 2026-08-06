import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
import time

from si_vq import si_vq,si2_vq
from si2_kmeans import si_kmeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


from scipy.fft import fft, fftfreq

# Dataframe has columns labeled testfold0, testfold1, ... 

def get_fold(df,test_fold,split,do_train=True):
    col = 'testfold'+str(split)
    if do_train:
        match = df[col] != test_fold
    else:
        match = df[col] == test_fold        
    return df[match]

def get_split_info(df):
    nfolds = int(df['testfold0'].max() + 1)
    nsplits = len([ 0 for x in df.columns if 'testfold' in x]) 
    return nfolds,nsplits 

def grab_windows(df,n_windows,window_length,just_indices=False,chan=0,do_train=True, rng=np.random.default_rng(0)):
    n_subjects = len(df.index)
    print(n_subjects)
    print(n_windows)
        
    n_per_subject = np.full(n_subjects,int(np.floor(n_windows/n_subjects)))
    how_many_get_1_more = n_windows-np.sum(n_per_subject)
    n_per_subject[:how_many_get_1_more] += 1
    n_per_subject = n_per_subject[rng.permutation(n_subjects)] # permute 

    print(np.sum(n_per_subject))
    windows = np.zeros((n_windows,window_length))
    indices = np.zeros((n_windows,3),dtype=int)
    window_index = 0
    i = -1
    for subject,row in df.iterrows():
        i += 1
        # make sample points and then map to valid windows in files (and exclude seizure times)
        if do_train:
            ends = row['file_train_end']   
            starts = [0 for _ in ends]
        else:
            starts = row['file_test_start']
            ends = row['file_lengths']

        file_subset_lengths = np.zeros(len(row['filepaths']),dtype=int)        
        for j,file in enumerate(row['filepaths']):       
            if starts[j] is None or ends[j] is None:
                continue                 
            
            if isinstance(row['exclusions'],list) and row['exclusions'][j] is not None:
                start = starts[j]
                for ind in row['exclusions'][j]:
                    if ind[1]<=start or ind[0]>=ends[j]:
                        continue
                    if ind[0]>ends[j]:
                        break                        
                    file_subset_lengths[j] += max(ind[0]-start-window_length+1,0)
                    start = ind[1] 
                    
                file_subset_lengths[j] += max(ends[j]-start-window_length+1,0)
            else:
                file_subset_lengths[j] += max(ends[j]-starts[j]-window_length+1,0)
        effective_length = np.sum(file_subset_lengths).astype(int)
        global_window_indices = np.sort(rng.choice(effective_length, size=n_per_subject[i], replace=False))  # these can be overlapping

        effective_ends = np.cumsum(file_subset_lengths)#
        effective_starts = np.concatenate(([0],effective_ends[:-1]))
 #       print(effective_starts)
  #      print(effective_ends)
        
        for j,file in enumerate(row['filepaths']):       
            if starts[j] is None or ends[j] is None:
                continue   
            # grab indices here
            global_subset = global_window_indices[(global_window_indices>=effective_starts[j]) & (global_window_indices<effective_ends[j])]
            if len(global_subset)==0:
                print('No windows in this file')
                continue
            window_starts = starts[j]+np.sort(rng.choice(file_subset_lengths[j], size=len(global_subset), replace=False))  # these can be overlapping

#            print('start',window_starts[0],'last start',window_starts[-1],'starts',starts[j],'ends',ends[j],'length',ends[j]-starts[j],'subset length', file_subset_lengths[j])
            if isinstance(row['exclusions'],list) and row['exclusions'][j] is not None:                
                for ind in row['exclusions'][j]:                    
                    start_match = (window_starts>=ind[0]) & (window_starts<ind[1])
                    end_match = (window_starts + window_length-1>=ind[0]) & (window_starts+ window_length-1<ind[1])
                    window_starts[ start_match|end_match  ] += ind[1]-ind[0] +window_length - 1 # move them forward need to check
                    
                for w_ind in window_starts: # sanity check loops
                    for ind in row['exclusions'][j]:                    
                        if (w_ind>= ind[0] and w_ind < ind[1]) or (w_ind +window_length-1 >= ind[0] and w_ind+window_length-1 < ind[1]):
                            print('window in exclusion',w_ind,w_ind+window_length-1,ind[0],ind[1])
                            print('old time',w_ind-(ind[1]-ind[0]),w_ind+window_length-1-(ind[1]-ind[0]))
            indices[window_index:window_index+len(window_starts),0] += i
            indices[window_index:window_index+len(window_starts),1] += j
            indices[window_index:window_index+len(window_starts),2] = window_starts
                                        

            if not just_indices:
                raw = mne.io.read_raw_edf(file, preload=True)                                
#                x = raw[chan, 0:window_starts[-1]+window_length][0].reshape(-1)
                x = raw[chan, :][0].reshape(-1)
                print(window_starts[-1]+window_length) #
                time_indices = np.arange(window_length)[np.newaxis,:]+window_starts[:,np.newaxis]
#                print(len(x),window_starts[-1]+window_length,np.max(time_indices))
                windows[window_index:window_index+len(window_starts)] = x[time_indices]
            else: # just indices
                for k in range(len(window_starts)):
                    assert window_starts[k]+window_length >= starts[j], '{} {} {} {}'.format(subject,j,window_starts[k],row['file_lengths'][j])               
                    assert window_starts[k]+window_length <= ends[j], '{} {} {} {}'.format(subject,j,window_starts[k],row['file_lengths'][j])               
            window_index+=len(window_starts)
    return windows, indices, df.index.to_list()

# get train folds and train time splits
def grab_windows_train(df,test_fold,split,n_windows,window_length,just_indices=False,chan=0,rng=np.random.default_rng(0)):
    return grab_windows(get_fold(df,test_fold,split,True),n_windows,window_length,just_indices,chan,True, rng)

# get train folds but valid time splits
def grab_windows_valid(df,test_fold,split,n_windows,window_length,just_indices=False,chan=0,rng=np.random.default_rng(0)):
    return grab_windows(get_fold(df,test_fold,split,True),n_windows,window_length,just_indices,chan,False, rng) 


# get test folds and test time splits
def grab_windows_test(df,test_fold,split,n_windows,window_length,just_indices=False,chan=0,rng=np.random.default_rng(0)):
    return grab_windows(get_fold(df,test_fold,split,False),n_windows,window_length,just_indices,chan,False, rng)    

# use_svd flag added ----------------
def create_codebooks(windows_train, n_centroids, centroid_len, use_sign_invariant=False, use_svd=False, do_sphere=False, tol_factor=1e-4):
    print(f"DEBUG: create_codebooks called with use_svd={use_svd}, do_sphere={do_sphere}")
    if do_sphere:
        windows_train /= np.sqrt(np.sum(windows_train**2,axis=1,keepdims=True))
    centroids, labels, shifts, _, inertia, n_iter = si_kmeans(
        windows_train.squeeze(), n_centroids, centroid_len, metric='cosine', use_sign_invariant=use_sign_invariant, use_svd=use_svd, do_sphere=do_sphere,
        init='random', n_init=1, tol=tol_factor*np.var(windows_train), rng=0,  verbose=True)

    return centroids,labels, shifts, inertia, n_iter # distances



def make_psds_shift(windows, psd_length, sampling_interval=None, n_shifts=0, rng=np.random.default_rng(0)):
    n_windows, window_length = windows.shape
    sample_index = np.arange(n_windows)
    if psd_length > window_length:
        # need windowing function
        windows_ = np.hstack((windows,np.zeros((n_windows,psd_length-window_length))))
    elif psd_length == window_length:
        windows_ = windows
    else:
        if n_shifts > 0:
            sample_index = np.tile(np.arange(n_windows)[:,np.newaxis],(1,n_shifts)).reshape((-1,1))
            print(sample_index)
            row_index = np.tile(sample_index,(1,psd_length))
            col_index = rng.choice(window_length-psd_length, size=n_windows*n_shifts)[:,np.newaxis]+np.arange(psd_length)[np.newaxis,:]
            windows_ = windows[row_index, col_index]
        elif n_shifts == -1:  # non-overlapping
            num_psds_per_window = window_length // psd_length
            trunc_window_size = num_psds_per_window * psd_length
            sample_index = np.tile(np.arange(n_windows)[:,np.newaxis],(1,num_psds_per_window)).reshape((-1,1))
            windows_ = windows[:, :int(trunc_window_size)].reshape((-1,psd_length))
        else:
            windows_ = windows[:, :psd_length]
    if psd_length % 2 == 0:  # even
        psd_pos_length = psd_length//2
    else:
        psd_pos_length = (psd_length+1)//2
    psd = np.abs(fft(windows_, axis=1)[:, :psd_pos_length])**2  # positive part of spectrum

    if sampling_interval is not None:
        freq = fftfreq(window_length, sampling_interval)[:psd_pos_length]
    else:
        freq = np.arange(psd_pos_length)
    return psd, sample_index.ravel(), freq


def make_psd_freq(windows, nfft=None, sampling_interval=None):
    n_windows, window_length = windows.shape
    if nfft is None:
        nfft = window_length
    if window_length % 2 == 0:  # even
        psd_pos_length = nfft//2
    else:
        psd_pos_length = (nfft+1)//2
    if sampling_interval is not None:
        freq = fftfreq(nfft, sampling_interval)[:psd_pos_length]
    else:
        freq = np.arange(psd_pos_length)
    return freq

def make_psds(windows, nfft=None):
    n_windows, window_length = windows.shape
    if nfft is None:
        nfft = window_length
    if nfft % 2 == 0:  # even
        psd_pos_length = nfft//2
    else:
        raise NotImplementedError('Odd length windows or NFFT is not implemented yet')
        psd_pos_length = (nfft+1)//2
    return np.abs(fft(windows, axis=1, n=nfft)[:, :psd_pos_length])**2  # positive part of spectrum
def psd_norm_sqrt(psds):
    x = np.sqrt(normalize(np.abs(psds), axis=1, norm='l1'))
    return x

def window2psd_norm_sqrt(windows,nfft=None):
    return psd_norm_sqrt(make_psds(windows,nfft))

def create_spectral_codebooks(windows_train, n_centroids, nfft=None):
    # spec_kmeans = Pipeline([
    #     ('psd', FunctionTransformer(window2psd_norm_sqrt,validate=True)),
    #     ('kmeans', KMeans(n_clusters=n_centroids, random_state=0, n_init=1, init='random'))
    # ])
    #  spec_kmeans = spec_kmeans.fit()
    # win2psd_transformer = FunctionTransformer(window2psd_norm_sqrt, validate=True)
    x = window2psd_norm_sqrt(windows_train,nfft)
    kmeans = KMeans(n_clusters=n_centroids, random_state=0, n_init=1, init='random').fit(x)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return kmeans, labels,  kmeans.inertia_, kmeans.n_iter_  # distances

def plot_codebooks(codebook, name,ncols=None,figsize=(9.5, 8)):
    n = codebook.shape[0]
    if ncols is None:
        if n>8:
            ncols = 8
        else:
            ncols = int(n/2)
    nrows = int(np.ceil(n/ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    for i in range(nrows):
        for j in range(ncols):
            if i*ncols+j < n:
                axs[i, j].plot(codebook[i * ncols + j])
            axs[i, j].axis('off')  # hide axis
    
    plt.suptitle(name, fontsize=16)
    plt.tight_layout()
    plt.show()
    return fig, axs


def get_spectral_counts_per_segment(segments, window_length, km_model, non_overlap=True):
    # make overlapping or non-overlapping windows 1
    n, segment_length = segments.shape
    K = km_model.n_clusters

    if non_overlap:
        num_windows_per_segment = segment_length // window_length
        trunc_signal_size = num_windows_per_segment * window_length
        windows = segments[:, :int(trunc_signal_size)].reshape((-1, window_length))
    else:
        new_offsets = np.arange(0, segment_length - window_length, window_length//2)
        array_of_window_indices = np.arange(window_length)[np.newaxis, :] + new_offsets[:, np.newaxis]
        num_windows_per_segment = len(array_of_window_indices)
        windows = np.vstack([segments[i][array_of_window_indices] for i in range(len(segments))])
    print(windows.shape)
    nfft_pos = km_model.cluster_centers_.shape[1]
    window_cluster_labels = km_model.predict(window2psd_norm_sqrt(windows, nfft=2*nfft_pos))
    segment_clusters = window_cluster_labels.reshape((n, num_windows_per_segment))
    counts = np.zeros((n, K))
    for i in range(n):
        counts[i] = np.bincount(segment_clusters[i], minlength=K)
    return counts, segment_clusters

def get_counts_per_segment(segments,window_length,centroids,use_sign_invariant=False,non_overlap=True):
    # make overlapping or non-overlapping windows 1
    n,segment_length = segments.shape
    K,centroid_length = centroids.shape
    
    if non_overlap:
        num_windows_per_segment = segment_length // window_length
        trunc_signal_size = num_windows_per_segment * window_length
        
        windows = segments[:,:int(trunc_signal_size)].reshape((-1,window_length))
    else:
        new_offsets = np.arange(0,segment_length-window_length,window_length-centroid_length+1)
        array_of_window_indices = np.arange(window_length)[np.newaxis,:]+new_offsets[:,np.newaxis]
        
        num_windows_per_segment = len(array_of_window_indices)
        
        windows = np.vstack([segments[i][array_of_window_indices] for i in range(len(segments))])

    print(windows.shape)

    if use_sign_invariant:
        window_cluster_labels, _, window_dists, window_signs = si2_vq(windows,centroids,'cosine')
    else:
        window_cluster_labels, _, window_dists = si_vq(windows,centroids,'cosine')

    segment_clusters = window_cluster_labels.reshape((n, num_windows_per_segment))
    counts = np.zeros((n, K))
    for i in range(n):
        counts[i] = np.bincount(segment_clusters[i], minlength=K)
    return counts, segment_clusters


def get_sex_labels(df, indices, index2name):
    sex_list = [df.loc[index2name[i], 'sex'] for i in range(len(index2name))]
    index2namesex = {i: n+s for i, (n, s) in enumerate(zip(index2name.values(), sex_list))}
    sex_labels = np.array([int(s == 'M') for s in sex_list])[indices]
    sex_keys = ['F', 'M']
    return sex_labels, sex_keys, index2namesex

def get_labels_subjects(segment_info):
    label_vector = np.vstack( [np.full((segment_info[s][0].shape[0],1),i)  for i,s in enumerate(segment_info)] ).ravel() 
    # this is by type, which can be collapsed to Het vs. WT and the 3 strains
    # to get sex requires using the subject files names segments[2] to go back to the dataframe 
    
    subject_vector = np.zeros_like(label_vector)
    # each key is a different class, we can create internal CV splits that hold out individuals 
    # need to stratify so that each fold has at least 1 individual per type 
    i_individuals = 0
    i_samples = 0
    
    subject_index2name = dict()
    for i,key in enumerate(segment_info):
        indices = segment_info[key][0]        
        subject_index = indices[:,0] # subject index relative to class
        subject_vector[i_samples:i_samples+len(subject_index)] = subject_index + i_individuals
        for j in range(len(subject_index)):
            si = int(subject_index[j])
            subject_index2name[si+i_individuals]=segment_info[key][1][si]
    
        i_individuals += int(np.max(subject_index)+1)
        i_samples += len(subject_index)
    return label_vector, subject_vector, subject_index2name



def get_counts_by_type_fold(df,test_fold,split,train_valid_test,n_segments,segment_length,window_length,waveform_dict,use_sign_invariant=False,use_spectral=False):
    grab_windows_funcs = {'train':grab_windows_train, 'test':grab_windows_test, 'valid':grab_windows_valid }
    grab_windows_func = grab_windows_funcs[train_valid_test]
    segments = dict()
    segment_info = dict()
    strains = df['strain'].unique()
    tscs = df['tsc'].unique()

    all_counts = dict()
    all_labels = dict()
    
    for strain in strains:
        for tsc in tscs:
            key_segment = (split,test_fold,strain,tsc)
            time_start= time.time()
#            signal_segments, segment_indices, name_list = grab_windows_func(df,test_fold,split,int(n_segments),segment_length)
            match = (df['tsc']==tsc) & (df['strain']==strain)            
            signal_segments, segment_indices, name_list = grab_windows_func(df[match],test_fold,split,int(n_segments),segment_length)
            segment_info[key_segment]=(segment_indices,name_list)            
            for key_dict in [(split,test_fold,strain2,bg2) for strain2 in strains for bg2 in tscs ]:
                centroids = waveform_dict[key_dict]
                print(signal_segments.shape)
                start_time = time.time()
                if use_spectral:
                    counts, segment_clusters = get_spectral_counts_per_segment(signal_segments,window_length,centroids)
                else:
                    counts, segment_clusters = get_counts_per_segment(signal_segments, window_length, centroids,
                                                                      use_sign_invariant)
                print("--- %s seconds ---" % (time.time() - start_time))
            
                print(segment_clusters.shape)
                all_labels[key_dict,key_segment] = segment_clusters 
                all_counts[key_dict,key_segment] = counts 

    return segment_info, all_counts, all_labels


def get_counts(df, test_fold, split, train_valid_test, n_segments, segment_length, window_length, waveform_dict,
               use_sign_invariant=False,use_spectral=False):
    grab_windows_funcs = {'train': grab_windows_train, 'test': grab_windows_test, 'valid': grab_windows_valid}
    grab_windows_func = grab_windows_funcs[train_valid_test]
    signal_segments, segment_indices, name_list = grab_windows_func(df, test_fold, split,
                                                                    int(n_segments), segment_length)
    labels = dict()
    counts = dict()
    strains = ('BXD87', 'DBA2', 'C57B6')
    tscs = ('Het', 'WT')
#    strains = list({k[2] for k in waveform_dict})
#    tscs = list({k[3] for k in waveform_dict})
#    print(strains,tscs)

    for key_dict in [(split, test_fold, strain2, bg2) for strain2 in strains for bg2 in tscs]:
        centroids = waveform_dict[key_dict]
        print(signal_segments.shape)
        start_time = time.time()
        if use_spectral:
            segment_counts, segment_clusters = get_spectral_counts_per_segment(signal_segments, window_length, centroids)
        else:
            segment_counts, segment_clusters = get_counts_per_segment(signal_segments, window_length, centroids,
                                                              use_sign_invariant)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(segment_clusters.shape)
        print(key_dict)
        labels[key_dict] = segment_clusters
        counts[key_dict] = segment_counts

    return segment_indices, name_list, counts, labels


def get_features(df, test_fold, split, train_valid_test, n_segments, segment_length):
    grab_windows_funcs = {'train': grab_windows_train, 'test': grab_windows_test, 'valid': grab_windows_valid}
    grab_windows_func = grab_windows_funcs[train_valid_test]
    signal_segments, segment_indices, name_list = grab_windows_func(df, test_fold, split,
                                                                    int(n_segments), segment_length)
    print(signal_segments.shape)
    start_time = time.time()
    # reshape `features` to (n_time_series, n_segments, n_features)
    sfreq = df['sfreq'].iloc[0] # assume all mice are the same, should be checked on any new dataset
    features = get_iclabel_features(signal_segments, sfreq=sfreq)
    print("--- %s seconds ---" % (time.time() - start_time))

    return segment_indices, name_list, features


# class_bg = lambda x: x[1]
# class_strain = lambda x: x[0]
# class_type = lambda x: x[0]+'_'+x[1]
# class_sex = lambda x: x[2]
# class_all = lambda x: x[0]+'_'+x[1]+'_'+x[2]

# classes_bg = list({class_bg(x) for x in gensex2subject})
# classes_strain = list({class_strain(x) for x in gensex2subject})
# classes_type = list({class_type(x) for x in gensex2subject})
# classes_sex = list({class_sex(x) for x in gensex2subject})
# classes_all = list({class_all(x) for x in gensex2subject})
