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
ENCODING_SCHEME = "shared"  #
# ENCODING_SCHEME = "all"

N_OBS = 5

ENCODING_SIZE = 5


LOGGING_LEVEL = 'INFO'

DATA_FNAME = "training-data-and-stats_{}_nobs_{}--{}.npy".format(
    ENCODING_SCHEME,
    N_OBS,
    hostname)


# hyperparams for shared, n_obs = 5; CVERR = 7.30% tpcxbb-dataset
# cverr results from 2020-05-30 with encoding size=5
if ENCODING_SCHEME == "shared" and N_OBS == 5:
    HYPER_PARAMS = {'ae_params': {'T': 1,
                                  'activation': 'relu',
                                  'batch_size': 128,
                                  'depth': 1,
                                  'early_stopping': True,
                                  'encoding_dim': 5,
                                  'lamda': 1,
                                  'learning_rate': 0.01,
                                  'n_epochs': 500,
                                  'nh': 100,
                                  'patience': 50},
                    'nn_params': {'input_shape': (17,), 'n_hidden_layers': 4, 'nh': 50}}

# *****************************************************************************************

# hyperparams for shared, n_obs = 1; CVERR = 9.15% tpcxbb-dataset
# cverr results from 2020-05-30 with encoding size=5
elif ENCODING_SCHEME == "shared" and N_OBS == 1:
    HYPER_PARAMS = {'ae_params': {'T': 2,
                                  'activation': 'relu',
                                  'batch_size': 128,
                                  'depth': 2,
                                  'early_stopping': True,
                                  'encoding_dim': 5,
                                  'lamda': 10,
                                  'learning_rate': 0.001,
                                  'n_epochs': 100,
                                  'nh': 20,
                                  'patience': 20},
                    'nn_params': {'input_shape': (17,), 'n_hidden_layers': 3, 'nh': 100}}


# hyperparams for all, n_obs=5 (cverr=7.58%) tpcxbb-dataset
# cverr results from 2020-05-30 with encoding size=5
elif ENCODING_SCHEME == "all" and N_OBS == 5:
    HYPER_PARAMS = {'ae_params': {'T': 2,
                                  'activation': 'relu',
                                  'batch_size': 128,
                                  'depth': 2,
                                  'early_stopping': True,
                                  'encoding_dim': 5,
                                  'lamda': 10,
                                  'learning_rate': 0.001,
                                  'n_epochs': 100,
                                  'nh': 20,
                                  'patience': 20},
                    'nn_params': {'input_shape': (17,), 'n_hidden_layers': 3, 'nh': 100}}


# hyperparams for all, n_obs=1 (cverr=12.69%) tpcxbb-dataset
# cverr results from 2020-05-30 with encoding size=5
elif ENCODING_SCHEME == "all" and N_OBS == 1:
    HYPER_PARAMS = {'ae_params': {'T': 2,
                                  'activation': 'relu',
                                  'batch_size': 128,
                                  'depth': 2,
                                  'early_stopping': True,
                                  'encoding_dim': 5,
                                  'lamda': 10,
                                  'learning_rate': 0.001,
                                  'n_epochs': 100,
                                  'nh': 20,
                                  'patience': 20},
                    'nn_params': {'input_shape': (17,), 'n_hidden_layers': 3, 'nh': 100}}


DEBUG_MODE = False


def get_config_dict():
    dico = {
        'N_WORKERS': N_WORKERS,
        'N_RUNS': N_RUNS,
        'SEED': SEED,
        'LODS_FOLDER_PATH': LODS_FOLDER_PATH,
        'LODS_FNAME': LODS_FNAME,
        'ENCODING_SCHEME': ENCODING_SCHEME,
        'N_OBS': N_OBS,
        'ENCODING_SIZE': ENCODING_SIZE,
        'LOGGING_LEVEL': LOGGING_LEVEL,
        'DATA_FNAME': DATA_FNAME,
        'HYPER_PARAMS': HYPER_PARAMS,
        'N_TRAIN_PER_JOB': N_TRAIN_PER_JOB,
        'N_SHARED_TRAIN_PER_JOB': N_SHARED_TRAIN_PER_JOB,
        'DEBUG_MODE': DEBUG_MODE

    }
    return dico
