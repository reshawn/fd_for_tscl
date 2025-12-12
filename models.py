
# import cuml.accel
# cuml.accel.install()

from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from gluonts.mx.model.transformer import TransformerEstimator
from gluonts.mx.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx.model.wavenet import WaveNetEstimator
from gluonts.mx.trainer.callback import TrainingHistory

from gluonts.mx.trainer import Trainer
RANDOM_STATE = 777


def load_model(model_name, params):
    '''
    Load model given text name
    '''
    history = TrainingHistory()
    if model_name == "ridge_classifier":        model = RidgeClassifier(alpha=params['alpha'], random_state=RANDOM_STATE)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth']) # random_state=RANDOM_STATE
    elif model_name == 'logistic_regression':
        model = LogisticRegression(random_state=RANDOM_STATE)
    elif model_name == 'transformer':
        model = TransformerEstimator(
              freq=params['freq'],
              context_length=params['context_length'],
              prediction_length=params['prediction_length'],
              inner_ff_dim_scale= params['inner_ff_dim_scale'],
              model_dim= params['model_dim'],
              embedding_dimension= params['embedding_dimension'],
              num_heads= params['num_heads'],
              dropout_rate= params['dropout_rate'],
              scaling=True, # True by default False
              trainer=Trainer(ctx=params['ctx'],epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                              num_batches_per_epoch=100, callbacks=[history]), #
          )
    elif model_name == 'ffn':
        model = SimpleFeedForwardEstimator(
            num_hidden_dimensions= params['num_hidden_dimensions'], #num_hidden_dimensions,
            prediction_length=params['prediction_length'],
            context_length=params['context_length'],
            batch_normalization=True,
            mean_scaling=True,
            trainer=Trainer(ctx=params['ctx'],epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                            num_batches_per_epoch=100, callbacks=[history]),
        )
    elif model_name == 'wavenet':
            model = WaveNetEstimator(
                freq=params['freq'],
                prediction_length=params['prediction_length'],
                trainer=Trainer(ctx=params['ctx'],epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                                    num_batches_per_epoch=100, callbacks=[history], add_default_callbacks=False),
        )
    return model, history

def load_model_params(model_name, trial):
    '''
    Load model hyperparameters given text name
    '''
    params = {}
    if model_name == "ridge_classifier":
        params['alpha'] = trial.suggest_float('alpha', 0.1, 10)
    elif model_name == 'random_forest':
        params['n_estimators'] = trial.suggest_int('n_estimators', 1, 50)
        params['max_depth'] = trial.suggest_int('max_depth', 2, 32)
    elif model_name == 'logistic_regression':
        pass

    if model_name == 'transformer':
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
    if model_name == 'ffn':
            return {
              "num_hidden_dimensions": [trial.suggest_int("hidden_dim_{}".format(i), 10, 100) for i in range(trial.suggest_int("num_layers", 1, 5))],
              "trainer:learning_rate": trial.suggest_loguniform("trainer:learning_rate", 1e-6, 1e-3),
              "trainer:epochs": trial.suggest_int("trainer:epochs", 10, 100),
            }
    if model_name == 'wavenet':
            return {
                "trainer:learning_rate": trial.suggest_loguniform("trainer:learning_rate", 1e-6, 1e-3),
                "trainer:epochs": trial.suggest_int("trainer:epochs", 10, 100),
            }


    
    params['RANDOM_STATE'] = RANDOM_STATE
    return params




