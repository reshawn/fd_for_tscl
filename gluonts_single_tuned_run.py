import os
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import time
import optuna
import argparse


from gluonts.evaluation import make_evaluation_predictions
from gluonts.mx.trainer import Trainer
from gluonts.mx.trainer.callback import TrainingHistory
from gluonts.mx.distribution import StudentTOutput, MultivariateGaussianOutput
from sklearn.metrics import mean_absolute_error
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from gluonts.mx.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx.model.transformer import TransformerEstimator
from gluonts.mx.model.deepar import DeepAREstimator 
from gluonts.mx.model.wavenet import WaveNetEstimator
from gluonts.mx.model.seq2seq import MQCNNEstimator

import matplotlib.pyplot as plt
import json
from pathlib import Path
from gluonts.dataset.split import split
from gluonts.dataset.common import ListDataset
import copy

from gluonts.evaluation import Evaluator
from gluonts.evaluation import make_evaluation_predictions

from gluonts_utils import series_to_gluonts_dataset, load_params
from fracdiff import invert_fd, invert_ffd

class Objective:
    def __init__( self, model, dataset_name, X, d=None, thresh=None):
        '''
        model: str
        dataset_name: str
        X: pd.DataFrame. Expects this df to have a target column called 'target', 
        and if using a frac diffed target to be inverted, also expects the original version to be named 'values_o'
        d: float. Differencing amount used if the target in X was frac diffed, None if none used.
        thresh: float. Threshold used if the target in X was frac diffed using the ffd method, None if none used or fd used.
        '''
        self.model = model
        self.dataset_name = dataset_name
        self.data_params = load_params('gluonts_params.txt', dataset_name)
        print(self.data_params)

        self.ctx = 'gpu(0)'
        self.original_value_col = 'target_o' # gets used in the inversion of fd if needed, 

        self.prediction_length = self.data_params['prediction_length']
        self.context_length = self.data_params['context_length']
        self.freq = self.data_params['freq']
        
        self.d = d
        self.thresh = thresh
        self.X = X
        X_train, X_test = X[:-self.prediction_length], X[-self.prediction_length:]
        X_train_val, X_val = X_train[:-self.prediction_length], X_train[-self.prediction_length:]
        
        self.tuning_dataset = series_to_gluonts_dataset(X_train_val, X_val,  self.data_params)
        self.eval_dataset =  series_to_gluonts_dataset(X_train, X_test,  self.data_params)
        

    def get_params(self, trial) -> dict:
        if self.model == 'feedforward':
            return {
              "num_hidden_dimensions": [trial.suggest_int("hidden_dim_{}".format(i), 10, 100) for i in range(trial.suggest_int("num_layers", 1, 5))],
              "trainer:learning_rate": trial.suggest_loguniform("trainer:learning_rate", 1e-6, 1e-3),
              "trainer:epochs": trial.suggest_int("trainer:epochs", 10, 100),
            }
        elif self.model == 'wavenet':
            return {
                "trainer:learning_rate": trial.suggest_loguniform("trainer:learning_rate", 1e-6, 1e-3),
                "trainer:epochs": trial.suggest_int("trainer:epochs", 10, 100),
            }
        elif self.model == 'mqcnn':
            return {
                "trainer:learning_rate": trial.suggest_loguniform("trainer:learning_rate", 1e-6, 1e-3),
                "trainer:epochs": trial.suggest_int("trainer:epochs", 10, 100),
            }
        elif self.model == 'deepar':
            return {
                "num_cells": trial.suggest_int("num_cells", 10, 100),
                "num_layers": trial.suggest_int("num_layers", 1, 10),
                "trainer:learning_rate": trial.suggest_loguniform("trainer:learning_rate", 1e-6, 1e-3),
                "trainer:epochs": trial.suggest_int("trainer:epochs", 10, 100)
            }
        elif self.model == 'transformer':
            # num_heads must divide model_dim
            valid_pairs = [ (i,d) for i in range(10,101) for d in range(1,11) if i%d == 0  ]
            model_dim_num_heads_pair = trial.suggest_categorical("model_dim_num_heads_pair", valid_pairs)

            return {
                "inner_ff_dim_scale": trial.suggest_int("inner_ff_dim_scale", 1, 5),
                "model_dim": model_dim_num_heads_pair[0],
                "embedding_dimension": trial.suggest_int("embedding_dimension", 1, 10),
                "num_heads": model_dim_num_heads_pair[1],
                "dropout_rate": trial.suggest_uniform("dropout_rate", 0.0, 0.5),
                "trainer:learning_rate": trial.suggest_loguniform("trainer:learning_rate", 1e-6, 1e-3),
                "trainer:epochs": trial.suggest_int("trainer:epochs", 10, 100),
            }
        
    def load_model(self, params):
        history = TrainingHistory()
        if self.model == 'feedforward':
            estimator = SimpleFeedForwardEstimator(
                num_hidden_dimensions= params['num_hidden_dimensions'], #num_hidden_dimensions,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                batch_normalization=False,
                mean_scaling=False,
                trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                                num_batches_per_epoch=100, callbacks=[history]),
            )
        elif self.model == 'wavenet':
            estimator = WaveNetEstimator(
                freq=self.freq,
                prediction_length=self.prediction_length,
                trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                                    num_batches_per_epoch=100, callbacks=[history], add_default_callbacks=False),
            )
        elif self.model == 'mqcnn':
            estimator = MQCNNEstimator(
                freq=self.freq,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                distr_output=StudentTOutput(),
                quantiles=None,
                scaling=False, 
                trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                                num_batches_per_epoch=100, callbacks=[history], hybridize=False),
            )
        elif self.model == 'deepar':
            estimator = DeepAREstimator(
                freq=self.freq,
                context_length=self.context_length,
                distr_output=StudentTOutput(),
                prediction_length=self.prediction_length,
                # num_cells= params['num_cells'],
                # num_layers= params['num_layers'],
                scaling=False, # True by default
                trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                                num_batches_per_epoch=100, callbacks=[history]),
            )
        elif self.model == 'transformer':
            estimator = TransformerEstimator(
                freq=self.freq,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                distr_output=StudentTOutput(),
                inner_ff_dim_scale= params['inner_ff_dim_scale'],
                model_dim= params['model_dim'],
                embedding_dimension= params['embedding_dimension'],
                num_heads= params['num_heads'],
                dropout_rate= params['dropout_rate'],
                # scaling=False, # True by default False
                trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                                num_batches_per_epoch=100, callbacks=[history]),
            )

        return estimator, history

    def train_test(self, params, tuning=True):
        model, history = self.load_model(params)

        if tuning:
            predictor = model.train(self.tuning_dataset.train, self.tuning_dataset.test)
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=self.tuning_dataset.test,
                predictor=predictor,
            )
        else:
            predictor = model.train(self.eval_dataset.train)
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=self.eval_dataset.test,
                predictor=predictor,
            )

        forecasts = list(forecast_it)
        tss = list(ts_it)
        # invert here if fd
        if self.d:
            if tuning:
                new_fd_series = self.tuning_dataset.train[0]['target'].copy()
                original_series = self.X[self.original_value_col][:-(2*self.prediction_length)].copy()
                actual = self.X[self.original_value_col][:-self.prediction_length].copy().values
            else:
                new_fd_series = self.eval_dataset.train[0]['target'].copy()
                original_series = self.X[self.original_value_col][:-self.prediction_length].copy()
                actual = self.X[self.original_value_col].copy().values

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
        
        print(f'#####__tuning rmse = {agg_metrics["RMSE"]}__###### ')
        return agg_metrics, predictor, history
        

    def __call__(self, trial):

        params = self.get_params(trial)

        agg_metrics, _, _ = self.train_test(params, tuning=True)

        return agg_metrics['RMSE']

def run(X, model, dataset_name, save_label, n_trials, n_repeats, d=None, thresh=None):
    start_time = time.perf_counter()

    # run tuning
    study = optuna.create_study(direction="minimize")
    obj = Objective(
        model=model,
        dataset_name=dataset_name,
        X=X,
        d=d,
        thresh=thresh
        )
    study.optimize(
        obj,
        n_trials=n_trials,
    )
    trial = study.best_trial

    # print results
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print(f'Runtime: {time.perf_counter() - start_time}')

    # unpack params for next runs
    if model == 'feedforward':
        trial.params["num_hidden_dimensions"] = [ trial.params[f"hidden_dim_{i}"] for i in range(trial.params["num_layers"]) ]
    elif model == 'transformer':
        trial.params["model_dim"] = trial.params["model_dim_num_heads_pair"][0]
        trial.params["num_heads"] = trial.params["model_dim_num_heads_pair"][1]

    # repeat best run 5 times
    mases = []
    smapes = []
    rmses = []
    params_sets = []
    save_dir = f'results/monash_gluonts_single_tuned_runs/{model}_{dataset_name}_{save_label}_{n_trials}_trials'
    os.makedirs(save_dir, exist_ok=True)
    for i in range(n_repeats):
        res, predictor, history = obj.train_test(trial.params, tuning=False)

        # plot and save training history
        plt.plot(history.loss_history, label='Training Loss')
        plt.plot(history.validation_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)

        # Save the figure
        plt.savefig(f'{save_dir}/learning_curve_{i}.png')
        # save the history values
        with open(f'{save_dir}/loss_history_{i}.json', "w") as f:
            json.dump(history.loss_history, f)
        # Clear the current figure
        plt.clf()
        mases.append(res['MASE'])
        smapes.append(res['sMAPE'])
        rmses.append(res['RMSE'])
        params_sets.append(trial.params)


    mase_mean = np.array(mases).mean()
    mase_std = np.std(np.array(mases))
    rmses_mean = np.array(rmses).mean()
    rmses_std = np.std(np.array(rmses))
    smape_mean = np.array(smapes).mean()
    smape_std = np.std(np.array(smapes))

    print(f'##### MASE MEAN: {mase_mean} MASE STD: {mase_std}')
    print(f'##### sMAPE MEAN: {smape_mean} sMAPE STD: {smape_std}')
    print(f'##### RMSE MEAN: {rmses_mean} RMSE STD: {rmses_std}')


    # save best params to json
    with open(f'{save_dir}/params.json', "w") as f:
        json.dump(trial.params, f)

    # save the last predictor
    os.makedirs(f'{save_dir}/predictor', exist_ok=True)
    predictor.serialize(Path(f"{save_dir}/predictor"))

    end_time = time.perf_counter()
    runtime = (end_time - start_time) / 60

    file_path = "gluonts_results.txt"
    with open(file_path, "a") as file:
        file.write(
            f'''\n\n\n\n {model} {n_trials} tuning trials {n_repeats} repeat test evals on {dataset_name} {save_label} data form, Runtime: {runtime} minutes 
            \n Prediction length: {obj.prediction_length}, Context length: {obj.context_length}
            \n Params: {trial.params}
            \n MASE MEAN: {mase_mean} MASE STD: {mase_std}
            \n sMAPE MEAN: {smape_mean} sMAPE STD: {smape_std}
            \n RMSE MEAN: {rmses_mean} RMSE STD: {rmses_std}
            '''
        )

if __name__ == '__main__':

    ######################### m4 series ##########################################
    # X = pd.read_csv('m4_1165_fd.csv') # original fd, default thresh, no skips, d=0.1
    X = pd.read_csv('m4_1165_ffd.csv') # fixed width fd, 0.01 thresh, d=0.95
    d=0.95
    thresh = 0.01
    X_withoutfd = X.drop(columns=['values_fd']).rename(columns={"values_o":"target"})
    X_fdtarget = X.rename(columns={"values_fd": "target"}) 
    # this X_fdtarget leaves o in fdr, fd_fdr is still the opposite to compare the impact of fd as target, 
    # and so far most tests have shown o+fd being more effective than fd alone
    X_otarget_fdinfdr = X.rename(columns={"values_o": "target"})
    save_labels = ['original_values', 'fd_target_values', 'fd_in_fdr']

    # run(X_withoutfd, 'transformer', 'm4_daily_dataset', save_label=save_labels[0], n_trials=15, n_repeats=5)
    # run(X_fdtarget, 'transformer', 'm4_daily_dataset', save_label=save_labels[1], n_trials=15, n_repeats=5, d=d, thresh=thresh)
    # run(X_otarget_fdinfdr, 'transformer', 'm4_daily_dataset', save_label=save_labels[2], n_trials=15, n_repeats=5)
    
    # run(X_withoutfd, 'feedforward', 'm4_daily_dataset', save_label=save_labels[0], n_trials=15, n_repeats=5)
    # run(X_fdtarget, 'feedforward', 'm4_daily_dataset', save_label=save_labels[1], n_trials=15, n_repeats=5, d=d, thresh=thresh)
    # run(X_otarget_fdinfdr, 'feedforward', 'm4_daily_dataset', save_label=save_labels[2], n_trials=15, n_repeats=5)
    
    # run(X_withoutfd, 'wavenet', 'm4_daily_dataset', save_label=save_labels[0], n_trials=15, n_repeats=5)
    # run(X_fdtarget, 'wavenet', 'm4_daily_dataset', save_label=save_labels[1], n_trials=15, n_repeats=5, d=d, thresh=thresh)
    # run(X_otarget_fdinfdr, 'wavenet', 'm4_daily_dataset', save_label=save_labels[2], n_trials=15, n_repeats=5)


    ########################## S&P 500 series ##########################################
    with open('/mnt/c/Users/resha/Documents/Github/balancing_framework/spy5m_bintp_labelled.pkl', 'rb') as f:
        df_original = pickle.load(f) # ohlv + transactions + labels + bintp labels
    with open('/mnt/c/Users/resha/Documents/Github/balancing_framework/spy5m_bintp004_episodes_fracdiff.pkl', 'rb') as f:
        df_fd = pickle.load(f) # fracdiffed ohlcv + transactions + labels
    with open('/mnt/c/Users/resha/Documents/Github/balancing_framework/spy5m_labelled_episodes_ta.pkl', 'rb') as f:
        df_ta = pickle.load(f) # ohlcv + ~120 TA features + labels
    with open('/mnt/c/Users/resha/Documents/Github/balancing_framework/spy5m_ta_fracdiff.pkl', 'rb') as f:
        df_fd_ta = pickle.load(f) # fracdiffed ohlcv + transactions + ~120 TA features + labels

    # original series target with rest of ohlcv
    df = df_ta[["volume", "vwap", "open", "close", "high", "low", "transactions"]] # 0.01 0.001
    X_withoutfd= df[:20_000].copy()
    X_withoutfd['target'] = X_withoutfd['close'].shift(-1)
    X_withoutfd.dropna(inplace=True)
    # fd all series, retaining all originals, fd target
    df = df_fd_ta[["volume", "vwap", "open", "close", "high", "low", "transactions"]]
    X_fdtarget = df[:20_000].copy()
    X_fdtarget = X_fdtarget.join(X_withoutfd.add_suffix(f'_o'), how='outer')
    X_fdtarget['target'] = X_fdtarget['close'].shift(-1)
    X_fdtarget.dropna(inplace=True)
    # same as above but with original target
    X_otarget_fdinfdr = X_fdtarget.copy()
    X_otarget_fdinfdr.rename(columns={"target": "target_fd", "target_o": "target"}, inplace=True)

    del df, df_fd, df_ta, df_fd_ta
    d = 0.2
    thresh = 1e-5
    save_labels = ['original_values', 'fd_target_values', 'fd_in_fdr']


    # run(X_withoutfd, 'transformer', 'sp500', save_label=save_labels[0], n_trials=15, n_repeats=5)
    # run(X_fdtarget, 'transformer', 'sp500', save_label=save_labels[1], n_trials=15, n_repeats=5, d=d, thresh=thresh)
    # run(X_otarget_fdinfdr, 'transformer', 'sp500', save_label=save_labels[2], n_trials=15, n_repeats=5)
    
    # run(X_withoutfd, 'feedforward', 'sp500', save_label=save_labels[0], n_trials=15, n_repeats=5)
    # run(X_fdtarget, 'feedforward', 'sp500', save_label=save_labels[1], n_trials=15, n_repeats=5, d=d, thresh=thresh)
    # run(X_otarget_fdinfdr, 'feedforward', 'sp500', save_label=save_labels[2], n_trials=15, n_repeats=5)
    
    run(X_withoutfd, 'wavenet', 'sp500', save_label=save_labels[0], n_trials=15, n_repeats=5)
    run(X_fdtarget, 'wavenet', 'sp500', save_label=save_labels[1], n_trials=15, n_repeats=5, d=d, thresh=thresh)
    run(X_otarget_fdinfdr, 'wavenet', 'sp500', save_label=save_labels[2], n_trials=15, n_repeats=5)
