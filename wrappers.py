import random
import time
import pandas as pd
import numpy as np

from rocket_functions import generate_kernels, apply_kernels
from fracdiff import frac_diff_bestd
from evaluator import Evaluator

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import coherence

def my_train_test_split(X,y,test_size=0.2, ignore_size=0.25, random_state=777):
    # a train test split that does not randomly pull from the first 25% of data
    # because frac diff will drop values and we want the same test sets for with and without frac diff
    random.seed(random_state)  # Set the random seed
    n = round(len(X)*test_size)  # Number of random numbers
    start = round(len(X)*ignore_size)  # Start of range
    end = len(X)-1  # End of range

    random_numbers = random.sample(range(start, end + 1), n)
    all_numbers = set(range(start, end + 1))
    remaining_numbers = list(all_numbers - set(random_numbers))

    X_train, X_test = X.iloc[remaining_numbers], X.iloc[random_numbers]
    y_train, y_test = y.iloc[remaining_numbers], y.iloc[random_numbers]

    return X_train, X_test, y_train, y_test

def train_test_split_by_indices(X,y,test_indices, num_dropped=0):
    test_indices = test_indices - num_dropped
    # print(f'test_indices: {test_indices.min()} - {test_indices.max()}')
    # print(f'X: {X.index.min()} - {X.index.max()}')
    # print(f'y: {y.index.min()} - {y.index.max()}')
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]
    X_train = X.drop(test_indices)
    y_train = y.drop(test_indices)
    return X_train, X_test, y_train, y_test



def run_measurements(X, y, chunk_size, dataset_name, model_name, start_chunk=0, end_chunk=-1, num_runs=10, frac_diff=False, rocket=False, d=None, thresh=None, gluonts=False):
    """
    Evaluate a model's performance on a dataset in terms of adaptation and consolidation measures.

    Parameters
    ----------
    X : pandas.DataFrame
        The features of the dataset
    y : pandas.Series
        The target of the dataset
    chunk_size : int
        The size of each chunk to split the data into
    cold_start_size : int
        The size of the cold start
    dataset_name : str
        The name of the dataset
    model_name : str
        The name of the model, from supported list in models.py, e.g: ['ridge_classifier', 'random_forest', 'logistic_regression']
    start_chunk, end_chunk : int
        Pair of index integers, defaults to full range of chunks, but allows for segmented runs with easier checkpointing
    num_runs : int
        The number of runs to perform the evaluation
    frac_diff : bool
        Whether to use fractional differencing as a preprocessing step
    rocket : bool
        Whether to use rocket as a preprocessing step
    d : float
        fractional differencing parameter, if frac_diff was done and needs to be inverted latre
    thresh : float
        fractional differencing threshold, if frac_diff was done and needs to be inverted later
    Returns
    -------
    adaptation_results : pandas.DataFrame
        The results of the adaptation measure
    consolidation_results : pandas.DataFrame
        The results of the consolidation measure
    prep_info : dict
        A dictionary containing any relevant preprocessing information
    """
    # PREPROCESSING -------------------------------------------------------------------------------------------------

    # Frac Diff
    # Included in the framework for completeness but its more efficient to do this outside of the framework, storing and loading since its slow
    if frac_diff:
        start = time.perf_counter()
        old_len = len(X)
        X, fd_change_pct = frac_diff_bestd(X)
        end = time.perf_counter()
        X.dropna(inplace=True)
        y = y.iloc[:len(X)]
        num_dropped = old_len - len(X)
        time_taken_mins = (end-start)/60

    # Rocket
    if rocket:
        input_length = X.shape[-1]
        kernels = generate_kernels(input_length, 10_000)
        start = time.perf_counter()
        X = apply_kernels(X.to_numpy(), kernels)
        end = time.perf_counter()
        time_taken_mins = (end-start)/60
    
    # Pack the preprocessing info
    prep_info = {}
    if frac_diff:
        prep_info['frac_diff'] = True
        prep_info['fd_change_pct'] = fd_change_pct
        prep_info['num_dropped'] = num_dropped
        prep_info['fd_time_taken_mins'] = time_taken_mins
    if rocket:
        prep_info['rocket'] = True
        prep_info['rocket_time_taken_mins'] = time_taken_mins

    # EVALUATION -------------------------------------------------------------------------------------------------

    eval = Evaluator(dataset_name, 
        model_name, 
        X, 
        y, 
        num_runs=num_runs, 
        start_chunk=start_chunk, 
        end_chunk=end_chunk, 
        chunk_size=chunk_size, 
        test_size=None, 
        d=d, 
        thresh=thresh
    )


    # in the case of frac diff or any rolling window method, remember to subtract the dropped rows from the cold start size
    print("RUNNING MEASUREMENTS")
    if gluonts:
        adaptation_results = eval.run_measurements_gluonts()
        consolidation_results = None
    else:
        adaptation_results, consolidation_results = eval.run_measurements()


    # PRINT VISUALIZATIONS -------------------------------------------------------------------------------------------------
    

    return adaptation_results, consolidation_results, prep_info




def viz(a_results, c_results, metric='f1', title='Original', dir=None):

    means1 = np.array([ x[f'{metric}_mean'] for x in a_results ])
    std1 = np.array([ x[f'{metric}_std'] for x in a_results ])
    means2 = np.array([ x[f'{metric}_mean'] for x in c_results ])
    std2 = np.array([ x[f'{metric}_std'] for x in c_results ])

    timestamps1 = np.array([ x['last_ts'] for x in a_results ])
    timestamps2 = np.array([ x['last_ts'] for x in c_results ])
    
    # print(f'means: {means1}, {means2}, std: {std1}, {std2}, timestamps: {timestamps1}, {timestamps2}')

    _viz(means1, means2, std1, std2, timestamps1, timestamps2, metric, title, dir)

def _viz(means1, means2, std_dev1, std_dev2, timestamps1, timestamps2, metric, title, dir):
    """
    Visualize two sets of increasing numbers with standard deviation shaded areas.
    
    Parameters
    ----------
    means1 : numpy.ndarray
        The means of the first set of numbers.
    means2 : numpy.ndarray
        The means of the second set of numbers.
    std_dev1 : numpy.ndarray
        The standard deviations of the first set of numbers.
    std_dev2 : numpy.ndarray
        The standard deviations of the second set of numbers.
    timestamps1 : numpy.ndarray
        The timestamps of the first set of numbers.
    timestamps2 : numpy.ndarray
        The timestamps of the second set of numbers.
    metric : str
        The metric being visualized.
    title : str
        The title of the plot.
    dir: str
        Directory to save the plots in.
    Returns
    -------
    None
    """
    x1 = timestamps1 #np.arange(len(means1))
    x2 = timestamps2 #np.arange(len(means2))
    # Generate coherence values
    # f, coherence_values = coherence(means1, means2, fs=1)
    
    plt.figure(figsize=(6, 4))  
    # Plot the sets using seaborn
    # ax1 = plt.subplot(2, 1, 1)
    sns.lineplot(x=x1, y=means1, label='Adaptation')
    sns.lineplot(x=x2, y=means2, label='Consolidation')
    # Add shaded faces for standard deviation
    plt.fill_between(x1, means1 - std_dev1, means1 + std_dev1, alpha=0.3, label='Adaptation Std Dev')
    plt.fill_between(x2, means2 - std_dev2, means2 + std_dev2, alpha=0.3, label='Consolidation Std Dev')

    plt.title(title)
    plt.xlabel('Timestamp')
    plt.ylabel(f'{metric}')
    plt.legend()

    # Plot the coherence values
    # ax2 = plt.subplot(2, 1, 2)
    # plt.plot(f, coherence_values, label='Coherence')
    # plt.xlabel('Frequency')
    # plt.ylabel('Coherence')
    # plt.legend()

    plt.tight_layout()  # adjust the layout to fit the figure size
    # plt.show()

    # save the figure
    plt.savefig(f'{dir}/viz_{title}_{metric}.png')




