import os
def prepare_sparse_train_set_window(path_to_csv_files, site_freq_path, 
                                    session_length=10, window_size=10):
    #''' ВАШ КОД ЗДЕСЬ'''
    #Load Vocab
    with open(site_freq_path, 'rb') as handle:
        vocab_site = pickle.load(handle)    
    
    ses_n = 0
    df_f = pd.DataFrame()
    
    for f in glob(os.path.join(path_to_csv_files, '*.csv')):
        df = pd.read_csv(f)

        #getting user_id
        fn = os.path.basename(f)
        user_id = int(re.search(r'\d{4}', fn).group())     

        for r in range(0, len(df), window_size):
            df_s = df.iloc[r:r+session_length].copy()['site'].value_counts().to_frame()
            df_s.reset_index(inplace=True)
            df_s.rename(columns={'site': 'site_freq', 'index': 'site'}, inplace=True)
            #df_s['site_id'] = site_vocab[df_s['site']][0]
            
            # df_s['site_id'] = df_s.apply(get_site_id, args=(vocab_site,), axis=1)
            vfunc = np.vectorize(get_site_id_v)
            df_s['site_id'] = vfunc(df_s['site'], vocab_site)
            
            df_s['session'] = ses_n
            df_s['user_id'] = user_id
            ses_n += 1
            
            df_f = df_f.append(df_s, ignore_index=True)     

    #print(df_f.shape)
    lil = lil_matrix((len(df_f['session'].unique()), len(vocab_site)))
    y = []

    for row in range(len(df_f['session'].unique())):
        df_s = df_f[df_f['session'] == row]
        y.append(df_s['user_id'].unique()[0]) #target

        for i, r in df_s.iterrows():
            lil[row, r['site_id']-1] = r['site_freq']
    return lil, y    

site_freq_path = os.path.join(PATH_TO_DATA,'site_freq_3users.pkl')
path_to_csv_files = os.path.join(PATH_TO_DATA,'3users')
prepare_sparse_train_set_window(site_freq_path, path_to_csv_files)