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
from gluonts_utils import series_to_gluonts_dataset

# Don't forget to point the output to a log file

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--forms', type=str, nargs='+', help="Data forms, one or more of: ['fd','fod','o'] corresponding to ['frac_diff' , 'first_order_diff', 'original']")
    parser.add_argument('-d','--dataset', type=str, help="Dataset, matching one pullable from gluonts")
    args = parser.parse_args()

    data_forms = [ f.lower() for f in args.forms ]
    dataset_name = args.dataset
    print(f'Running measurements with params: format={data_forms}, dataset_name={dataset_name}')

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
        if forecast_horizon is None: 
            forecast_horizon = 24
        dataset = monash_df_to_gluonts_train_datasets(loaded_data, frequency, forecast_horizon)

    if dataset is None:
        if dataset_name in dataset_names:
            dataset = get_dataset(dataset_name)
        else:
            raise ValueError(f"Dataset {dataset_name} not found in gluonts availables or local monash files.")
        
    series_num = -1
    series_to_focus_on = 5771 #1165
    

    for entry in tqdm(dataset.test):
        # series_num += 1
        # # for using a single pre-picked series
        # if series_num > series_to_focus_on:
        #     break
        # if series_num != series_to_focus_on:
        #     continue

        ####################################### Prep Feats & Labels  ####################################################################################

        # used for producing balanced labels
        def adjust_split_point(series, threshold=0.05, max_iterations=1000):
            split_point = np.median(series)
            increment = 0.01  # Initial increment
            iterations = 0

            while iterations < max_iterations:
                binary_series = (series > split_point).astype(int)
                balance = binary_series.mean()[0]

                # Check if balance is within the threshold
                if abs(balance - 0.5) <= threshold:
                    break

                # Adjust increment based on how far off the balance is
                distance_from_balance = abs(balance - 0.5)
                increment = max(0.001, increment * (1 + distance_from_balance))  # Scale increment

                # Adjust split point based on balance
                if balance < 0.5:
                    split_point -= increment
                else:
                    split_point += increment

                iterations += 1

            return split_point

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
            # df['difference'] = df['values'].diff()  # Difference from the previous value
            # df['percentage_change'] = df['values'].pct_change()  # Percentage change from the previous value
            df['cumulative_sum'] = df['values'].cumsum()  # Cumulative sum
            df['cumulative_mean'] = df['values'].expanding().mean()  # Cumulative mean
            df['cumulative_max'] = df['values'].cummax()  # Cumulative maximum
            df['cumulative_min'] = df['values'].cummin()  # Cumulative minimum
            df.dropna(inplace=True)

            return df.reset_index(drop=True)

        df = pd.DataFrame(entry['target'])
        series = df[[0]].copy()
        changes = series.diff().dropna()

        split_point = adjust_split_point(changes)
        binary_series = (changes > split_point).astype(int)

        df = df.iloc[:-1].copy() # reflect diff drop of row 1 on original df
        df['label'] = binary_series.values
        df.rename(columns={0: 'values'}, inplace=True)

        df = derive_feats(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()


        ####################################### Prep Data Forms ##############################################################################################


        for data_forms in form_sets:
            print(f'Data forms: {data_forms}')      
              
            X = pd.DataFrame()
            y = df['label']

            for data_form in data_forms:
                if data_form == 'fd':
                    df_fd, fd_change_pct, fd_params = frac_diff_bestd(df.drop(columns=['label']), type='ffd' )
                    print(f'frac diff params: {fd_params}, change_pct: {fd_change_pct}')
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


            model_name = 'random_forest'
            # lower bounds to chunk size
            # min to run, chunk_size > X.shape(1) * 10
            # min for 3 splits, chunk_size > X.shape(0) /3 ;;; can adjust this min, unsure atm how many series fails condition
            # upper bound to chunk size
            # max num of splits, 10, 10 > (X.shape(0) / chunk_size) therefore default to 10% chunk size, using the lower bound
            curr_split_pct = 0.1
            chunk_size = int(X.shape[0] * curr_split_pct)
            model_min_chunk_size = X.shape[1] * 10
            max_split_pct = 0.33
            curr_splits = X.shape[0] / chunk_size
            while chunk_size < model_min_chunk_size:
                curr_split_pct += 0.01
                if curr_split_pct > max_split_pct:
                    print(f'series length {X.shape[0]} with {X.shape[1]} feats does not meet criteria of >=10*n_features and a resulting split % > max_split_pct={max_split_pct*100}%.')
                    # raise Exception(f'series length {X.shape[0]} with {X.shape[1]} feats does not meet criteria of >=10*n_features and a resulting split % > max_split_pct={max_split_pct*100}%.')
                    break
                chunk_size = int(X.shape[0] * curr_split_pct)

            # cold_start_size = 10_000
            num_runs = 10

            print(f'Running measurements with params: format={data_form},chunk_size={chunk_size}, num_runs={num_runs}, \
                dataset_name={dataset_name}, model_name={model_name}')

            a,c,p = run_measurements(X, y, chunk_size, dataset_name, model_name, num_runs=num_runs, frac_diff=False)

            ####################################### Save Results and Visualizations ###############################################################################

            import os
            base_dir = 'results/monash_runs'
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

            viz(a, c, metric='accuracy', title=data_form, dir=save_dir) 
            viz(a, c, metric='f1', title=data_form, dir=save_dir) 

            print(f'Finished measurements with params: format={data_forms},chunk_size={chunk_size}, num_runs={num_runs}, \
                dataset_name={dataset_name}, model_name={model_name}')