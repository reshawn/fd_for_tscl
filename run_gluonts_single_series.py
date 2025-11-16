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

    model_name = 'transformer'


    valid_forms = ['fd' , 'fod', 'o', 'ta_o', 'ta_fod', 'ta_frac_diff', 'auto']
    for data_form in data_forms:
        if data_form not in valid_forms:
            raise ValueError(f'The data_form arg must be one of: {valid_forms}')

    if data_forms[0] == 'auto': form_sets = [['o', 'fd' , 'fod'], ['o', 'fd'], ['o', 'fod'], ['o']] 
    else: form_sets = [data_forms]


    start = time.time()


    ####################################### Load Data ##############################################################################################

    # print(f"Available datasets: {dataset_names}")
    dataset = None
    monash_dir = "monash_data"

    if os.path.exists(f"{monash_dir}/{dataset_name}.tsf"):
        loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(f"{monash_dir}/{dataset_name}.tsf")
        if forecast_horizon is None: forecast_horizon = 24
        dataset = monash_df_to_gluonts_train_datasets(loaded_data, frequency, forecast_horizon)

    if dataset is None:
        if dataset_name in dataset_names:
            dataset = get_dataset(dataset_name)
        else:
            raise ValueError(f"Dataset {dataset_name} not found in gluonts availables or local monash files.")
        

    series_num = -1
    series_to_focus_on = 1165



    for entry in tqdm(dataset.test):
        series_num += 1
        # for using a single pre-picked series
        if series_num > series_to_focus_on:
            break
        if series_num != series_to_focus_on:
            continue

        ####################################### Prep Feats & Labels  ####################################################################################

        def derive_feats(df):
            # Define window sizes for rolling calculations
            window_sizes = [2, 3, 4, 5]

            # Calculate rolling window features for different window sizes
            for window_size in window_sizes:
                df[f'rolling_mean_{window_size}'] = df['values'].rolling(window=window_size).mean()
                df[f'rolling_sum_{window_size}'] = df['values'].rolling(window=window_size).sum()
                df[f'rolling_min_{window_size}'] = df['values'].rolling(window=window_size).min()
                df[f'rolling_max_{window_size}'] = df['values'].rolling(window=window_size).max()
                df[f'rolling_std_{window_size}'] = df['values'].rolling(window=window_size).std()

            # Calculate additional features
            df['lag_1'] = df['values'].shift(1)  # Previous value
            df['lag_2'] = df['values'].shift(2)  # Value two steps back
            df['lag_3'] = df['values'].shift(3)  # Value three steps back
            df['lag_4'] = df['values'].shift(4)  # Value four steps back
            df['difference'] = df['values'].diff()  # Difference from the previous value
            df['percentage_change'] = df['values'].pct_change()  # Percentage change from the previous value
            df['cumulative_sum'] = df['values'].cumsum()  # Cumulative sum
            df['cumulative_mean'] = df['values'].expanding().mean()  # Cumulative mean
            df['cumulative_max'] = df['values'].cummax()  # Cumulative maximum
            df['cumulative_min'] = df['values'].cummin()  # Cumulative minimum
            df.dropna(inplace=True)

            return df.reset_index(drop=True)
        
        df = pd.DataFrame(entry['target'])
        df['label'] = 0
        df.rename(columns={0: 'values'}, inplace=True)

        # df = derive_feats(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()



        ####################################### Prep Data Forms ##############################################################################################


        for data_forms in form_sets:
            print(f'Data forms: {data_forms}')      
                
            X = pd.DataFrame()
            y = df['label']

            for data_form in data_forms:
                if data_form == 'fd':
                    df_fd, fd_change_pct, fd_params = frac_diff_bestd(df.drop(columns=['label']), type='ffd')
                    # pull the params used for the target column's transform
                    d, thresh = fd_params['values']
                    df_fd.dropna(inplace=True)
                    df_fd.reset_index(drop=True, inplace=True)
                    X = X.join(df_fd.add_suffix(f'_{data_form}'), how='outer')
                    del df_fd

                elif data_form == 'o':
                    X = X.join(df.drop(columns=['label']).add_suffix(f'_{data_form}'), how='outer')
                
                elif data_form == 'fod':
                    # if wanted to omit cols from diff
                    # diff = df.drop(['volume', 'transactions', 'label'], axis=1).diff()
                    # diff = diff.join(df[['volume', 'transactions']])
                    diff = df.drop(['label'], axis=1).diff().add_suffix(f'_{data_form}')
                    X = X.join(diff, how='outer')
                # del df
                # elif data_form == 'ta_original':
                #     X = X.join(df_ta.drop(columns=['label']).add_suffix(f'_{data_form}'), how='outer')
                #     # del df_ta
                # elif data_form == 'ta_fod':
                #     diff = df_ta.drop(['label'], axis=1).diff().add_suffix(f'_{data_form}')
                #     X = X.join(diff, how='outer')
                #     del df_ta
                # elif data_form == 'ta_frac_diff':
                #     X = X.join(df_fd_ta.drop(columns=['label']).add_suffix(f'_{data_form}'), how='outer')
                #     del df_fd_ta
            
            X.dropna(inplace=True)
            # drop y with index not in X
            y = y[y.index.isin(X.index)]

            # X.to_csv(f'X_{data_form}.csv')
            
            # exit()

            ####################################### Run Framework ##############################################################################################

            # lower bounds to chunk size
            # min to run, chunk_size > X.shape(1) * 10
            # min for 10 splits, chunk_size > X.shape(0) /10 ;;; can adjust this min, unsure atm how many series fails condition
            # upper bound to chunk size
            # max num of splits, 100, 100 > (X.shape(0) / chunk_size) therefore default to 1% chunk size, using the lower bound
            
            chunk_size = int(X.shape[0] * 0.1)
            min_splits = 3
            min_chunk_size = X.shape[1] * 10
            if chunk_size < min_chunk_size:
                chunk_size = min_chunk_size
            if chunk_size > (X.shape[0] / min_splits):
                print(f'series length {X.shape[0]} with {X.shape[1]} feats does not meet criteria of >=10*n_features and a resulting count of splits > min_splits={min_splits}.')
                pass
            # cold_start_size = 10_000
            num_runs = 1

            print(f'Running measurements with params: format={data_form},chunk_size={chunk_size}, num_runs={num_runs}, \
                dataset_name={dataset_name}, model_name={model_name}')
            print(X.columns)
            
            a,c,p = run_measurements(X, y, chunk_size, dataset_name, model_name, num_runs=num_runs, frac_diff=False)

            ####################################### Save Results and Visualizations ###############################################################################

            import os
            base_dir = '/mnt/c/Users/resha/Documents/Github/balancing_framework/results/monash_runs'
            dir_ext = f'{dataset_name}_{series_num}_{data_forms}_chunk_size={chunk_size}_num_runs={num_runs}_{model_name}'
            save_dir = os.path.join(base_dir, dir_ext)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            with open(f'{save_dir}/adaptation_results.pkl', 'wb') as f:
                pickle.dump(a, f)
            with open(f'{save_dir}/consolidation_results.pkl', 'wb') as f:
                pickle.dump(c, f)

            end = time.time()
            print(f"Runtime: {(end - start) / 60} minutes")

            viz(a, c, metric='mase', title=data_form, dir=save_dir) 
            viz(a, c, metric='rmse', title=data_form, dir=save_dir) 

            print(f'Finished measurements with params: format={data_forms},chunk_size={chunk_size}, num_runs={num_runs}, \
                dataset_name={dataset_name}, model_name={model_name}')