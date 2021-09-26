import numpy as np
import pandas as pd
from glob import glob
import os
import pickle
from scipy.sparse import csr_matrix, lil_matrix
from scipy import stats
from matplotlib import pyplot as plt
import re
from tqdm import tqdm #tqdm_notebook, tnrange
import itertools
from datetime import datetime
import line_profiler

def get_site_id_v(col1, vocab_site):
    return vocab_site[col1][0]

def prepare_sparse_train_set_window(path_to_csv_files, site_freq_path,
                                    session_length=10, window_size=10):
    #''' ВАШ КОД ЗДЕСЬ'''
    #Load Vocab
    with open(site_freq_path, 'rb') as handle:
        vocab_site = pickle.load(handle)

    ses_n = 0
    np_f = np.zeros(4)

    for f in tqdm(glob(os.path.join(path_to_csv_files, '*.csv'))):
        #f_np = np.genfromtxt(f, delimiter=';')
        df = pd.read_csv(f) #getting user data
        f_np = df.to_numpy()

        #getting user_id
        fn = os.path.basename(f)
        user_id = int(re.search(r'\d{4}', fn).group())

        for r in range(0, len(f_np), window_size):
            s = f_np[r:r+session_length, 1]
            sites , counts = np.unique(s, return_counts=True)
            s = np.concatenate((sites.reshape(-1,1), counts.reshape(-1,1)), axis=1)
            s = np.concatenate((s, np.array([user_id for i in range(len(s))]).reshape(-1,1)), axis=1)
            s = np.concatenate((s, np.array([ses_n for i in range(len(s))]).reshape(-1,1)), axis=1)

            np_f = np.vstack([np_f, s])
            ses_n += 1

    np_f = np.delete(np_f, 0, 0)
    vfunc = np.vectorize(get_site_id_v)
    np_f[:,0] = vfunc(np_f[:,0], vocab_site)


    lil = lil_matrix(( len(np.unique(np_f[:,3])), len(vocab_site))) # size: [sessions x sites]
    y = []


    first_y = True
    cur_ses = 0
    for r in np_f:
        lil[r[3], r[0]-1] = r[1]

        if cur_ses != r[3] or first_y:
            y.append(r[2])
            cur_ses = r[3]
            first_y = False

    return lil, y

def main():
    PATH_TO_DATA = 'capstone_user_identification'
    lil, y = prepare_sparse_train_set_window(os.path.join(PATH_TO_DATA,'10users'),
                                            os.path.join(PATH_TO_DATA,'site_freq_10users.pkl'),
                                            session_length=15, window_size=10)

if __name__ == "__main__":
    main()