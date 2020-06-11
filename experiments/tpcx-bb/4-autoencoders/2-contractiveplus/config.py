import socket
hostname = socket.gethostname()

SEED = 20

N_WORKERS = 10

N_RUNS = 10

N_TRAIN_PER_JOB = -1

N_SHARED_TRAIN_PER_JOB = -1

assert N_SHARED_TRAIN_PER_JOB != 0
assert N_TRAIN_PER_JOB != 0

# path to config that generated the lods
LODS_FNAME = "lods_mul.bin"


LODS_FOLDER_PATH = "../../../../preprocessing/tpcx-bb/output/"


# Observation Scheme: shared or all
# ENCODING_SCHEME = "shared"  #
ENCODING_SCHEME = "all"

N_OBS = 5


LOGGING_LEVEL = 'INFO'

DATA_FNAME = "training-data-and-stats_{}_nobs_{}--{}.npy".format(
    ENCODING_SCHEME,
    N_OBS,
    hostname)


# 'encoding_dim': 12+ENCODING_SIZE   <--- for the auto-encoder
# 'input_shape': (12+ENCODING_SIZE,) <--- for neural network

# hyperparams for shared, n_obs = 5; CVERR = 10.97% (encoding_dim=10+12)
# cverr results from 2020-06-05
if ENCODING_SCHEME == "shared" and N_OBS == 5:
    HYPER_PARAMS = {'ae_params': {'activation': 'relu',
                                  'depth': 1,
                                  'early_stopping': True,
                                  'encoding_dim': 22,
                                  'gamma': 10,
                                  'lamda': 0,
                                  'learning_rate': 0.0001,
                                  'n_epochs': 1000,
                                  'nh': 50,
                                  'patience': 50},
                    'nn_params': {'input_shape': (22,), 'n_hidden_layers': 4, 'nh': 50}}


# hyperparams for shared, n_obs = 1; CVERR = 19.66% (encoding_dim=10+12)
# cverr results from 2020-06-05
elif ENCODING_SCHEME == "shared" and N_OBS == 1:
    HYPER_PARAMS = {'ae_params': {'activation': 'relu',
                                  'depth': 1,
                                  'early_stopping': True,
                                  'encoding_dim': 22,
                                  'gamma': 10,
                                  'lamda': 0,
                                  'learning_rate': 0.0001,
                                  'n_epochs': 1000,
                                  'nh': 50,
                                  'patience': 50},
                    'nn_params': {'input_shape': (22,), 'n_hidden_layers': 4, 'nh': 50}}


# hyperparams for all, n_obs=5 (cverr= 32.88%) (encoding_dim=10+12)
# cverr results from 2020-06-05
elif ENCODING_SCHEME == "all" and N_OBS == 5:
    HYPER_PARAMS = {'ae_params': {'activation': 'relu',
                                  'config_vec_size': 12,
                                  'depth': 2,
                                  'early_stopping': True,
                                  'encoding_dim': 22,
                                  'gamma': 10,
                                  'input_dim': 286,
                                  'lamda': 0.01,
                                  'learning_rate': 0.001,
                                  'n_epochs': 1000,
                                  'nh': 100,
                                  'patience': 50},
                    'nn_params': {'input_shape': (22,), 'n_hidden_layers': 4, 'nh': 20}}


# hyperparams for all, n_obs=1 (cverr= 55.99%) (encoding_dim=5+12)
# cverr results from 2020-06-01
elif ENCODING_SCHEME == "all" and N_OBS == 1:
    HYPER_PARAMS = {'ae_params': {'activation': 'sigmoid',
                                  'depth': 2,
                                  'early_stopping': True,
                                  'encoding_dim': 17,
                                  'gamma': 0.1,
                                  'lamda': 10,
                                  'learning_rate': 0.01,
                                  'n_epochs': 1000,
                                  'nh': 50,
                                  'patience': 50},
                    'nn_params': {'input_shape': (17,), 'n_hidden_layers': 2, 'nh': 20}}


DEBUG_MODE = False


def get_config_dict():
    dico = {
        'SEED': SEED,
        'LODS_FOLDER_PATH': LODS_FOLDER_PATH,
        'LODS_FNAME': LODS_FNAME,
        'ENCODING_SCHEME': ENCODING_SCHEME,
        'N_OBS': N_OBS,
        'LOGGING_LEVEL': LOGGING_LEVEL,
        'DATA_FNAME': DATA_FNAME,
        'HYPER_PARAMS': HYPER_PARAMS,
        'N_TRAIN_PER_JOB': N_TRAIN_PER_JOB,
        'N_SHARED_TRAIN_PER_JOB': N_SHARED_TRAIN_PER_JOB,
        'DEBUG_MODE': DEBUG_MODE

    }
    return dico
