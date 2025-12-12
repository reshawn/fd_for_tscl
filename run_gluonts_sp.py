from gluonts.dataset.repository import get_dataset, dataset_names
from gluonts.dataset.util import to_pandas

import pickle
import pandas as pd
import numpy as np
import json
import time
import argparse
from tqdm import tqdm

from wrappers import run_measurements, viz
from fracdiff import frac_diff_bestd
from monash_data_utils import convert_tsf_to_dataframe, monash_df_to_gluonts_train_datasets
import os

# Don't forget to point the output to a log file

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--forms', type=str, nargs='+', help="Data forms, one or more of: ['fd','fod','o'] corresponding to ['frac_diff' , 'first_order_diff', 'original']")
    parser.add_argument('-d','--dataset', type=str, help="Dataset, matching one pullable from gluonts")
    args = parser.parse_args()

    data_forms = [ f.lower() for f in args.forms ]
    dataset_name = args.dataset
    print(f'Running measurements with params: format={data_forms}, dataset_name={dataset_name}')

    model_name = 'wavenet'
    original_target = True


    valid_forms = ['fd' , 'fod', 'o', 'ta_o', 'ta_fod', 'ta_frac_diff', 'auto']
    for data_form in data_forms:
        if data_form not in valid_forms:
            raise ValueError(f'The data_form arg must be one of: {valid_forms}')

    if data_forms[0] == 'auto': form_sets = ['o'] # [['o', 'fd']] # ['o'] #
    else: form_sets = [data_forms]


    start = time.time()


    ####################################### Load Data ##############################################################################################

    # with open('/mnt/c/Users/resha/Documents/Github/balancing_framework/spy5m_bintp_labelled.pkl', 'rb') as f:
    #     df_original = pickle.load(f) # ohlv + transactions + labels + bintp labels
    with open('/mnt/c/Users/resha/Documents/Github/balancing_framework/spy5m_bintp004_episodes_fracdiff.pkl', 'rb') as f:
        df_fd = pickle.load(f) # fracdiffed ohlcv + transactions + labels
    with open('/mnt/c/Users/resha/Documents/Github/balancing_framework/spy5m_labelled_episodes_ta.pkl', 'rb') as f:
        df_ta = pickle.load(f) # ohlcv + ~120 TA features + labels
    with open('/mnt/c/Users/resha/Documents/Github/balancing_framework/spy5m_ta_fracdiff.pkl', 'rb') as f:
        df_fd_ta = pickle.load(f) # fracdiffed ohlcv + transactions + ~120 TA features + labels
    
    df = df_ta[["volume", "vwap", "open", "close", "high", "low", "transactions"]] # 0.01 0.001
    df_fd = df_fd[["volume", "vwap", "open", "close", "high", "low", "transactions"]]
    # df

    with open('/mnt/c/Users/resha/Documents/Github/balancing_framework/spy5m_bintp_labelled.pkl', 'rb') as f:
        df_original = pickle.load(f) # ohlv + transactions + labels + bintp labels
    with open('/mnt/c/Users/resha/Documents/Github/balancing_framework/spy5m_fd.pkl', 'rb') as f:
        df_fd = pickle.load(f) # d=0.25 when using this one, post h=l drop
    # with open('/mnt/c/Users/resha/Documents/Github/balancing_framework/spy5m_bintp004_episodes_fracdiff.pkl', 'rb') as f:
    #     df_fd = pickle.load(f) # fracdiffed ohlcv + transactions + labels

    # filter single order bars
    df_original.drop(df_original[df_original['high'] == df_original['low']].index, inplace=True)

    d = 0.25 # 0.2 0.25
    thresh = 1e-5

    df_size = 20_000


    ####################################### Prep Data Forms ##############################################################################################
    for data_forms in form_sets:

        print(f'Data forms: {data_forms}')  
        X = pd.DataFrame()


        if 'fd' in data_forms:
            X = X.join(df_fd.add_suffix(f'_fd'), how='outer')
        del df_fd
        if 'o' in data_forms:
            X = X.join(df.add_suffix(f'_o'), how='outer')
        if 'fod' in data_forms:
            # if wanted to omit cols from diff
            # diff = df.drop(['volume', 'transactions', 'label'], axis=1).diff()
            # diff = diff.join(df[['volume', 'transactions']])
            diff = df.diff().add_suffix(f'_fod')
            X = X.join(diff, how='outer')
        del df
        if 'tao' in data_forms:
            X = X.join(df_ta.add_suffix(f'_tao'), how='outer')
        if 'tafod' in data_forms:
            diff = df_ta.diff().add_suffix(f'_tafod')
            X = X.join(diff, how='outer')
        del df_ta
        if 'tafd' in data_forms:
            X = X.join(df_fd_ta.add_suffix(f'_tafd'), how='outer')
        del df_fd_ta
        
        X.dropna(inplace=True) 
        y = pd.DataFrame([0] * len(X), index=X.index, columns=['label']) # dummy y for compatibility

        if original_target:
            X['target'] = X['close_o'].shift(-1)
        else: # fd target
            X['target_o'] = X['close_o'].shift(-1) # for keeping with consistent testing in o+fd, and also for convenient use in inverting the fd preds
            X['target'] = X['close_fd'].shift(-1)
        X.dropna(inplace=True)

            

        ####################################### Run Framework ##############################################################################################
        X = X[20000:120_000]
        chunk_size = 20_000 #int(X.shape[0] * 0.1)
        num_runs = 10

        print(f'Running measurements with params: format={data_form},chunk_size={chunk_size}, num_runs={num_runs}, \
            dataset_name={dataset_name}, model_name={model_name}')
        print(X.columns)

            # Check for any existing results
        start_chunk = 0 # start from beginning or last saved chunk
        import os
        base_dir = 'results/'
        series_num = 1
        run_title = 'run5'
        subfolder = f'{dataset_name}_{run_title}_{series_num}_{data_forms}_chunk_size={chunk_size}_num_runs={num_runs}_{model_name}'
        save_dir = os.path.join(base_dir, subfolder)
        os.makedirs(save_dir, exist_ok=True)
        potential_ar_res_file = f'{save_dir}/adaptation_results.pkl'
        potential_cr_res_file = f'{save_dir}/consolidation_results.pkl'
        ar_sofar = None
        cr_sofar = None
        if os.path.exists(potential_ar_res_file):
            with open(f'{save_dir}/adaptation_results.pkl', 'rb') as f:
                ar_sofar = pickle.load(f)
                if (start_chunk <= len(ar_sofar)-1) or (start_chunk == None):
                    start_chunk = len(ar_sofar)
                    print(f'Entered start index was already ran, using new start index {start_chunk}')
        if os.path.exists(potential_cr_res_file):
            with open(f'{save_dir}/consolidation_results.pkl', 'rb') as f:
                cr_sofar = pickle.load(f)
            
        if original_target:
            a,c,p = run_measurements(X, y, chunk_size, dataset_name, model_name, start_chunk=start_chunk, end_chunk=-1, num_runs=num_runs, gluonts=True)
        else: #fd target
            a,c,p = run_measurements(X, y, chunk_size, dataset_name, model_name, start_chunk=start_chunk, end_chunk=-1, num_runs=num_runs, d=d , thresh=thresh, gluonts=True )

        ####################################### Save Results and Visualizations ###############################################################################

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open(f'{save_dir}/adaptation_results.pkl', 'wb') as f:
            pickle.dump(a, f)
        # with open(f'{save_dir}/consolidation_results.pkl', 'wb') as f:
        #     pickle.dump(c, f)

        end = time.time()
        runtime = (end - start) / 60
        print(f"Runtime: {runtime} minutes")

        # viz(a, c, metric='mase', title=data_forms, dir=save_dir) 
        # viz(a, c, metric='rmse', title=data_forms, dir=save_dir) 
        print(a,c,p)
        print(f'Finished measurements with params: format={data_forms},chunk_size={chunk_size}, num_runs={num_runs}, \
            dataset_name={dataset_name}, model_name={model_name}')
        