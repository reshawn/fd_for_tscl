import numpy as np
import pandas as pd
import time
import os
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm
import warnings
import json
warnings.filterwarnings('ignore')

import optuna
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score

from sklearn.mixture import GaussianMixture


from gluonts_utils import series_to_gluonts_dataset, load_params
from gluonts.evaluation import Evaluator
from gluonts.evaluation import make_evaluation_predictions

RANDOM_STATE = 777
num_kernels = 10_000

from models import load_model, load_model_params
from fracdiff import invert_fd, invert_ffd

class Trainer:
    def __init__(self, model_name, dataset_name, num_runs=10, d=None, thresh=None):
        self.model_name = model_name
        self.tuned_params = None
        self.dataset_name = dataset_name
        self.num_runs = num_runs
        self.d = d
        self.thresh = thresh
        self.ctx = 'gpu(0)'
        self.gluonts_metric_type = 'RMSE'

        sklearn_models = ['ridge_classifier', 'random_forest', 'logistic_regression']
        gluonts_models = ['transformer']
        pytorch_models = []
        try:
            self.data_params = load_params('gluonts_params.txt', self.dataset_name)
        except Exception as e:
            print(e)
            if model_name in gluonts_models:
                print('Gluonts model selected, but no params for the dataset in gluonts_params.txt')
            self.data_params = None

        if model_name in sklearn_models:
            self.model_type = 'sklearn'
        elif model_name in pytorch_models:
            self.model_type = 'pytorch'
        elif model_name in gluonts_models:
            self.model_type = 'gluonts'
        else:
            raise ValueError(f"Model {model_name} not supported")

    def tune(self, X_train, y_train, X_val, y_val):
        # tune hyper params
        def objective(trial):
            if self.model_type == 'sklearn':
                params = load_model_params(self.model_name, trial)
                model, _ = load_model(self.model_name, params)
                model.fit(X_train, y_train)
                
                return accuracy_score(y_val, model.predict(X_val))
            elif self.model_type == 'pytorch':
                pass
            elif self.model_type == 'gluonts':
                params = load_model_params(self.model_name, trial)
                gluonts_dataset = series_to_gluonts_dataset(X_train, X_val,  self.data_params)
                for key in  self.data_params.keys():
                    params[key] =  self.data_params[key]
                params['ctx'] = self.ctx
                model, _ = load_model(self.model_name, params)
                predictor = model.train(gluonts_dataset.train, gluonts_dataset.test) # test in this case is the val set
                forecast_it, ts_it = make_evaluation_predictions(
                    dataset=gluonts_dataset.test,
                    predictor=predictor,
                )
                forecasts = list(forecast_it)
                tss = list(ts_it)
                if self.d:
                    new_fd_series = X_train['target'].copy()
                    original_series = X_train['target_o'].copy()
                    X = pd.concat([X_train, X_val], axis=0)
                    actual = X['target_o'].values

                    # iteratively invert the predicted fd target values, and prepare to repack the forecasts as expected by gluonts evaluator
                    unfd_samples = []
                    for predsampleset in forecasts[0].samples:
                        unfd_sampleset = []
                        fd = new_fd_series.copy()
                        o = original_series.copy()
                        for pred in predsampleset:
                            fd = np.append(fd, pred)
                            if self.thresh: ufd = invert_ffd(pd.Series(fd), o, self.d, self.thresh)
                            else: ufd = invert_fd(pd.Series(fd), o, self.d)
                            unfd_pred = ufd.iloc[-1]
                            unfd_sampleset.append(unfd_pred)
                            o = pd.Series(np.append(o, unfd_pred))
                        unfd_samples.append([unfd_sampleset])
                    forecasts[0].samples = np.array(unfd_samples).squeeze()
                    tss[0][0] = actual

                evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
                agg_metrics, item_metrics = evaluator(tss, forecasts)
                print(agg_metrics)
                return agg_metrics[self.gluonts_metric_type]
            else:
                raise ValueError(f"Model {self.model_name} not supported")

        time_a = time.perf_counter()
        if self.model_type == 'sklearn':
            study = optuna.create_study(direction='maximize')
        elif self.model_type == 'pytorch':
            study = optuna.create_study(direction='maximize')
        elif self.model_type == 'gluonts':
            study = optuna.create_study(direction='minimize')
        else:
            raise ValueError(f"Model {self.model_name} not supported")
        study.optimize(objective, n_trials=self.num_runs)
        time_b = time.perf_counter()
        # print the optimization time in minutes
        print(f"Optimization Time: {(time_b - time_a) / 60} minutes")

        # load best params
        best_params = study.best_params
        # add data params to params obj
        if self.model_type == 'gluonts':
            for key in  self.data_params.keys():
                best_params[key] =  self.data_params[key]
            best_params['ctx'] = self.ctx
            if self.model_name == 'transformer':
                best_params['model_dim'] = best_params['model_dim_num_heads_pair'][0]
                best_params['num_heads'] = best_params['model_dim_num_heads_pair'][1]

        self.tuned_params = best_params
        return best_params

    def train_eval(self, X_train, y_train, X_test, y_test):
        
        # train the model with the best params 10 times and store the mean and std accuracy, along with training and test times
        accs = []
        mccs = []
        f1s = []
        mapes, mases, smapes, maes, rmses = [], [], [], [], []
        train_times = []
        test_times = []
        print('Running with tuned params:',self.tuned_params)
        for i in range(self.num_runs):
            # -- training ----------------------------------------------------------
            time_a = time.perf_counter()
            model, history = load_model(self.model_name, self.tuned_params)
            sklearn_models = ['ridge_classifier', 'random_forest', 'logistic_regression']
            pytorch_models = []
            gluonts_models = ['transformer']
            if self.model_name in sklearn_models:
                model.fit(X_train, y_train)
            elif self.model_name in pytorch_models:
                pass
            elif self.model_name in gluonts_models:
                gluonts_dataset = series_to_gluonts_dataset(X_train, X_test,  self.data_params)
                model = model.train(gluonts_dataset.train)
                os.makedirs(f'results/gluonts_training_history/{self.dataset_name}', exist_ok=True)
                with open(f'results/gluonts_training_history/{self.dataset_name}/{self.model_name}_repeatrun_loss_history_{i}.json', "w") as f:
                    json.dump(history.loss_history, f)
            else:
                raise ValueError(f"Model {self.model_name} not supported")
            time_b = time.perf_counter()
            train_times.append(time_b - time_a)
            # -- test --------------------------------------------------------------
            time_a = time.perf_counter()

            if self.model_name in sklearn_models:
                acc = accuracy_score(y_test, model.predict(X_test))
                mcc = matthews_corrcoef(y_test, model.predict(X_test))
                f1 = f1_score(y_test, model.predict(X_test)) # , average='weighted' ; consider if setup changes to the more realistic multi-class imbalanced labels
                time_b = time.perf_counter()
                test_times.append(time_b - time_a)
                print(f"Run {i} Accuracy: {acc:.4f}")
                accs.append(acc)
                # mccs.append(mcc)
                f1s.append(f1)
            elif self.model_name in pytorch_models:
                pass
            elif self.model_name in gluonts_models:
                forecast_it, ts_it = make_evaluation_predictions(
                    dataset=gluonts_dataset.test,
                    predictor=model,
                )
                forecasts = list(forecast_it)
                tss = list(ts_it)
                if self.d:
                    new_fd_series = X_train['target'].copy()
                    original_series = X_train['target_o'].copy()
                    X = pd.concat([X_train, X_test], axis=0)
                    actual = X['target_o'].values
                    print(len(actual), len(tss[0]))

                    # iteratively invert the predicted fd target values, and prepare to repack the forecasts as expected by gluonts evaluator
                    unfd_samples = []
                    for predsampleset in forecasts[0].samples:
                        unfd_sampleset = []
                        fd = new_fd_series.copy()
                        o = original_series.copy()
                        for pred in predsampleset:
                            fd = np.append(fd, pred)
                            if self.thresh: ufd = invert_ffd(pd.Series(fd), o, self.d, self.thresh)
                            else: ufd = invert_fd(pd.Series(fd), o, self.d)
                            unfd_pred = ufd.iloc[-1]
                            unfd_sampleset.append(unfd_pred)
                            o = pd.Series(np.append(o, unfd_pred))
                        unfd_samples.append([unfd_sampleset])
                    forecasts[0].samples = np.array(unfd_samples).squeeze()
                    tss[0][0] = actual

                evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
                agg_metrics, item_metrics = evaluator(tss, forecasts)
                mapes.append(agg_metrics['MAPE'])
                mases.append(agg_metrics['MASE'])
                smapes.append(agg_metrics['sMAPE'])
                rmses.append(agg_metrics['RMSE'])

        result = {
            "accuracy_mean": np.mean(accs),
            # "mcc_mean": np.mean(mccs),
            "f1_mean": np.mean(f1s),
            "accuracy_std": np.std(accs),
            # "mcc_std": np.std(mccs),
            "f1_std": np.std(f1s),
            "mape_mean": np.mean(mapes),
            "mape_std": np.std(mapes),
            "rmse_mean": np.mean(rmses),
            "rmse_std": np.std(rmses),
            "mase_mean": np.mean(mases),
            "mase_std": np.std(mases),
            "smape_mean": np.mean(smapes),
            "smape_std": np.std(smapes),
            "time_training_seconds": np.mean(train_times),
            "time_test_seconds": np.mean(test_times),
            "model_name": self.model_name,
        }
        return result


# below functions are old, not in use, kept just in case
    def run(self, X, y, mode='normal', results=pd.DataFrame(), test_size=None):
        # Split the data
        # Because we are trying methods that drop rows, we need to make sure the test set is the same
        # Since its time series also better to preserve order
        if test_size is not None:
            X_test, y_test = (X[-test_size:], y[-test_size:])
            X, y = (X[:-test_size], y[:-test_size])
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        else:
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=RANDOM_STATE)

        self.tune(X_train, y_train, X_val, y_val)

        result = self.train_eval(X_train, y_train, X_test, y_test)

        result["mode"] = mode
        result["dataset_name"] = self.dataset_name
        # add result to results dataframe
        results = pd.concat([results,pd.DataFrame(result, index=[0])], ignore_index=True)
        
        return self.tuned_params, results

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def gen_chunk_base(df: pd.DataFrame, sample_size: int):
    """
    Generate a sample of data from a given dataframe using a Gaussian Mixture Model.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to generate a sample from.
    sample_size : int
        The number of samples to generate.

    Returns
    -------
    X : pd.DataFrame
        The generated sample without the label.
    y : pd.Series
        The generated sample's label.
    """
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df)
    # Split data into training and validation sets
    X_train, X_val = train_test_split(X, test_size=0.3, random_state=RANDOM_STATE)

    # Define the range of components to test
    n_components_range = range(1, 11)

    # Lists to store BIC and AIC scores
    bic_scores = []
    aic_scores = []
    gmms = []

    # Fit GMM models for each number of components and calculate BIC and AIC
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=RANDOM_STATE)
        gmm.fit(X_train)
        gmms.append(gmm)
        bic_scores.append(gmm.bic(X_val))
        aic_scores.append(gmm.aic(X_val))

    # Select the model with the lowest BIC
    optimal_components = n_components_range[np.argmin(bic_scores)]
    print(f"Optimal number of components according to BIC: {optimal_components}")

    gmm = gmms[optimal_components - 1]
    sample = gmm.sample(sample_size)[0]
    sample = scaler.inverse_transform(sample)
    sample = pd.DataFrame(sample, columns=df.columns)
    y = sample['label']
    # enforce a few simple logical rules on the generated sample
    y = y.round().abs()
    X = sample.drop('label', axis=1)
    X['high'] = X[['open', 'close', 'high', 'low']].max(axis=1)
    X['low'] = X[['open', 'close', 'high', 'low']].min(axis=1)

    return X,y

def gen_chunk(df: pd.DataFrame, sample_size: int, split_by_label: bool = False):
    # While inheriting labels by splitting the df by class and using separate GMMs:
    """
    Generate a sample chunk from the given DataFrame using Gaussian Mixture Models (GMMs).
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    sample_size (int): The desired size of the sample to generate.
    split_by_label (bool): If True, the DataFrame is split by class label and separate 
                           GMMs are used for each class to generate samples; otherwise,
                           a single GMM is used for the entire DataFrame.
    
    Returns:
    tuple: A tuple containing the generated features (X) and labels (y) as pandas DataFrames.
    """
    if split_by_label:
        sample_X_slices = []
        sample_y_slices = []
        for label in df.label.unique():
            df2 = df[df.label == label]
            sample_X_slice, _ = gen_chunk_base(df2, int(sample_size/len(df.label.unique())))
            sample_X_slices.append(sample_X_slice)
            sample_y_slices.append(pd.Series([label] * len(sample_X_slice)) ) # fill with repeated label to inherit from the class split
        X = pd.concat(sample_X_slices, axis=0)
        y = pd.concat(sample_y_slices, axis=0)
    # While Attempting to generate labels:
    else:
        X,y = gen_chunk_base(df, sample_size)
    
    return X,y
