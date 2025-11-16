import pickle
import pandas as pd
import numpy as np
import json
import time
import argparse

from wrappers import run_measurements, viz


# Don't forget to point the output to a log file

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--forms', type=str, nargs='+', help="Data forms, one or more of: ['fd' , 'fod', 'o']")
    parser.add_argument('-s', '--start_chunk', type=int, help="Start Chunk", default=0)
    parser.add_argument('-e', '--end_chunk', type=int, help="End Chunk", default=-1)
    args = parser.parse_args()

    data_forms = [ f.lower() for f in args.forms ]
    start_chunk = args.start_chunk
    end_chunk = args.end_chunk

    valid_forms = ['fd' , 'fod', 'o', 'tao', 'tafod', 'tafd']
    for data_form in data_forms:
        if data_form not in valid_forms:
            raise ValueError(f'The data_form arg must be one of: {valid_forms}')
    

    start = time.time()

    ####################################### Load Data ##############################################################################################

    with open('spy5m_bintp_labelled.pkl', 'rb') as f:
        df_original = pickle.load(f)
    # with open('spy5m_bintp004_episodes_fracdiff.pkl', 'rb') as f:
    #     df_fd = pickle.load(f)
    with open('spy5m_smallta.pkl', 'rb') as f:
        df_ta = pickle.load(f)
        df_ta['label'] = df_original['tp_0.004'][df_ta.index]
    with open('spy5m_smallta_fracdiff.pkl', 'rb') as f:
        df_fd_ta = pickle.load(f)
        df_fd_ta['label'] = df_original['tp_0.004'][df_ta.index]
    # PZ algorithm has some look ahead so remove the episode labels if those are there, will be uesd only for some kind of analysis afterwards
    # df = df_original.drop(columns=['episode']) 
    df_fd = df_fd_ta.copy()
    df_fd = df_fd[["volume", "vwap", "open", "close", "high", "low", "transactions"]]
    df_fd['label'] = df_original['tp_0.004'][df_fd.index]
    df = df_original[["volume", "vwap", "open", "close", "high", "low", "transactions", "tp_0.004"]].rename(columns={"tp_0.004": "label"}) # 0.01 0.001
    # df


    ####################################### Run Framework ##############################################################################################
    X = pd.DataFrame()
    y = df['label']


    if 'fd' in data_forms:
        X = X.join(df_fd.drop(columns=['label']).add_suffix(f'_fd'), how='outer')
    del df_fd
    if 'o' in data_forms:
        X = X.join(df.drop(columns=['label']).add_suffix(f'_o'), how='outer')
    if 'fod' in data_forms:
        # if wanted to omit cols from diff
        # diff = df.drop(['volume', 'transactions', 'label'], axis=1).diff()
        # diff = diff.join(df[['volume', 'transactions']])
        diff = df.drop(['label'], axis=1).diff().add_suffix(f'_fod')
        X = X.join(diff, how='outer')
    del df
    if 'tao' in data_forms:
        X = X.join(df_ta.drop(columns=['label']).add_suffix(f'_tao'), how='outer')
    if 'tafod' in data_forms:
        diff = df_ta.drop(['label'], axis=1).diff().add_suffix(f'_tafod')
        X = X.join(diff, how='outer')
    del df_ta
    if 'tafd' in data_forms:
        X = X.join(df_fd_ta.drop(columns=['label']).add_suffix(f'_tafd'), how='outer')
    del df_fd_ta
    
            
            
    X.dropna(inplace=True)
    # drop y with index not in X
    y = y[y.index.isin(X.index)]



    dataset_name = 'sp500'
    model_name = 'random_forest'
    chunk_size = 10_000
    # cold_start_size = 10_000
    num_runs = 10

    print(f'Running measurements with params: format={data_forms},chunk_size={chunk_size}, num_runs={num_runs}, \
        dataset_name={dataset_name}, model_name={model_name}')

    # Check for any existing results
    import os
    base_dir = 'results/'
    subfolder = f"chunk_size={chunk_size} num_runs={num_runs} {dataset_name} {model_name}/{data_forms}"
    save_dir = os.path.join(base_dir, subfolder)
    os.makedirs(save_dir, exist_ok=True)
    potential_ar_res_file = f'{save_dir}/adaptation_results.pkl'
    potential_cr_res_file = f'{save_dir}/consolidation_results.pkl'
    ar_sofar = None
    cr_sofar = None
    if os.path.exists(potential_ar_res_file):
        with open(f'{save_dir}/adaptation_results.pkl', 'rb') as f:
            ar_sofar = pickle.load(f)
            if start_chunk <= len(ar_sofar)-1:
                start_chunk = len(ar_sofar)
                print(f'Entered start index was already ran, using new start index {start_chunk}')
    if os.path.exists(potential_cr_res_file):
        with open(f'{save_dir}/consolidation_results.pkl', 'rb') as f:
            cr_sofar = pickle.load(f)


    a,c,p = run_measurements(X,
                            y,
                            chunk_size,
                            dataset_name,
                            model_name,
                            start_chunk=start_chunk,
                            end_chunk=end_chunk, # end inclusive
                            num_runs=num_runs,
                            frac_diff=False)
    
    if ar_sofar:
        a = ar_sofar + a
    if cr_sofar:
        c = cr_sofar + c


    ####################################### Save Results and Visualizations ###############################################################################

    
    with open(f'{save_dir}/adaptation_results.pkl', 'wb') as f:
        pickle.dump(a, f)
    with open(f'{save_dir}/consolidation_results.pkl', 'wb') as f:
        pickle.dump(c, f)

    end = time.time()
    print(f"Runtime: {(end - start) / 60} minutes")

    viz(a, c, metric='accuracy', title=data_forms, dir=save_dir) 
    viz(a, c, metric='f1', title=data_forms, dir=save_dir) 

