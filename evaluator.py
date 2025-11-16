import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split

from training import Trainer
from gluonts_utils import load_params

RANDOM_STATE = 777
num_kernels = 10_000


class Evaluator:
    def __init__(self, dataset_name, model_name, X, y, start_chunk=0, end_chunk=-1, num_runs=10, chunk_size=50000, test_size=None, d=None, thresh=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_runs = num_runs
        self.test_size = test_size
        self.d = d
        self.thresh = thresh
        self.start_chunk = start_chunk
        self.end_chunk = end_chunk
        self.results = pd.DataFrame()
        self.best_params = None

        self.split_at_ends = False
        self.data_params = None
        self.model_params = None
        self.skip_tuning = False

        
        gluonts_models = ['transformer']
        if self.model_name in gluonts_models:
            self.split_at_ends = True
            self.data_params = load_params('gluonts_params.txt', self.dataset_name)
            self.model_params = load_params('gluonts_model_params.txt', self.model_name)
            self.prediction_length = self.data_params['prediction_length']
            self.context_length = self.data_params['context_length']
        

        self.chunk_size = chunk_size
        # split df into chunks of size chunk_size and add 10% of each chunk into a list of test sets
        # NOTE: splitting it this way breaks the sequence. For 1 step predictions its okay but if treating X as a sequence:
            # would need to split by taking the ends of the chunks or something, maybe back to the GMMs since consol would need the ends
        self.chunks = []
        self.test_sets = []
        fin = False
        for i in range(0, len(X), chunk_size):
            if len(X)-i < chunk_size:
                break
            if len(X.iloc[i+chunk_size:i+chunk_size*2]) < chunk_size:
                X_tmp, y_tmp = X.iloc[i:], y.iloc[i:] 
            else:
                X_tmp, y_tmp = X.iloc[i:i+chunk_size], y.iloc[i:i+chunk_size]

            if self.split_at_ends:
                X_chunk, X_test = X_tmp.iloc[:-self.prediction_length], X_tmp.iloc[-self.prediction_length:]
                y_chunk, y_test = y_tmp.iloc[:-self.prediction_length], y_tmp.iloc[-self.prediction_length:]
            else:
                X_chunk, X_test, y_chunk, y_test = train_test_split(X_tmp, y_tmp, test_size=0.1, random_state=42)
            self.chunks.append((X_chunk, y_chunk))
            self.test_sets.append((X_test, y_test))
        self.num_chunks = len(self.chunks)



    def validation_set_split(self, X, y):
        if self.split_at_ends:
            X_train, X_val, y_train, y_val = X[:-self.prediction_length], X[-self.prediction_length:], y[:-self.prediction_length], y[-self.prediction_length:]
            return X_train, X_val, y_train, y_val
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
            return X_train, X_val, y_train, y_val
        

    
    
    def run_measurements(
            self,
            ):
        """
        Evaluate a model's adaptation performance on a dataset in an online setup.

        Parameters
        ----------

        Returns
        -------
        list
            A list of dictionaries containing the evaluation results for each chunk.
            
        The adaptation measure simulates an online learning scenario where the dataset
        is processed in chunks. For each chunk, the model is tuned and trained using
        seen data, and evaluated on a test set. The process is repeated for all chunks,
        and the results are returned as a list.
        """

        ar_results = []
        cr_results = []
        X_seen, y_seen = pd.DataFrame(), pd.DataFrame()
        trainer = Trainer(self.model_name, self.dataset_name, num_runs=self.num_runs, d=self.d, thresh=self.thresh)

        if (self.end_chunk == -1) or (self.end_chunk > len(self.chunks)-1): self.end_chunk=len(self.chunks)-1
        
        print(f'Running measurements from chunk {self.start_chunk} to {self.end_chunk} of {len(self.chunks)}')
        for i in tqdm(range(0, self.end_chunk+1), total=self.end_chunk - self.start_chunk + 1):
            chunk = self.chunks[i]
            X_chunk, y_chunk = chunk
            # X_seen, y_seen = pd.DataFrame(), pd.DataFrame()
            X_seen, y_seen = pd.concat([X_seen, X_chunk]), pd.concat([y_seen, y_chunk])
            y_seen = y_seen.squeeze()
            if i < self.start_chunk:
                continue
            
            print(f'Running measurement pair {i} of {self.num_chunks-1} chunks')
            X_train, X_val, y_train, y_val = self.validation_set_split(X_seen, y_seen)
            print(f'Tuning run {i} of {self.num_chunks-1} chunks')
            trainer.tune(X_train, y_train, X_val, y_val)

            print(f'AR Training run {i} of {self.num_chunks-1} chunks')
            X_chunk_test = self.test_sets[i][0]
            y_chunk_test = self.test_sets[i][1]
            result = trainer.train_eval(X_seen, y_seen, X_chunk_test, y_chunk_test)
            result['last_ts'] = X_seen.index[-1]
            ar_results.append(result)

            if i == 0:
                continue # no test sets besides that of the current chunk, so skip consolidation for first chunk
            print(f'CR Training run {i} of {self.num_chunks-1}')
            X_test = pd.concat([ temp[0] for temp in self.test_sets[:i] ]) # i+1
            y_test = pd.concat([ temp[1] for temp in self.test_sets[:i] ])

            result = trainer.train_eval(X_seen, y_seen, X_test, y_test)
            result['last_ts'] = X_seen.index[-1]
            cr_results.append(result)

        return ar_results, cr_results


