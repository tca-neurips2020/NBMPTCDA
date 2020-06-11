import socket
hostname = socket.gethostname()

SEED = 10

N_WORKERS = 10

N_RUNS = 10

N_TRAIN_PER_JOB = -1

N_SHARED_TRAIN_PER_JOB = -1

assert N_SHARED_TRAIN_PER_JOB != 0

assert N_TRAIN_PER_JOB != 0

# path to config that generated the lods
LODS_FNAME = "lods_mul.bin"

LODS_FOLDER_PATH = "../../../../preprocessing/tpcx-bb/output/"


# ENCODING_SCHEME = "shared"  #
ENCODING_SCHEME = "all"


N_OBS = 1

ENCODING_SIZE = 5


LOGGING_LEVEL = 'INFO'

DATA_FNAME = "training-data-and-stats_{}_nobs_{}--{}.npy".format(
    ENCODING_SCHEME,
    N_OBS,
    hostname)


# hyperparams for shared, n_obs = 5; CVERR = 7.41% (2020-05-30), encoding_size=5 (+12 for knobs)
if ENCODING_SCHEME == "shared" and N_OBS == 5:
    HYPER_PARAMS = {'encoder_params': {'_nh': 50,
                                       '_nhlayers': 1,
                                       'alpha': 0.01,
                                       'early_stopping': True,
                                       'gamma': 0.001,
                                       'lamda': 0.001,
                                       'learning_rate': 0.001,
                                       'n_epochs': 50,
                                       'patience': 5},
                    'nn_params': {'input_shape': (17,), 'n_hidden_layers': 3, 'nh': 50}}


# hyperparams for shared, n_obs = 1; CVERR = 8.74% (2020-05-30), encoding_size=5 (+12 for knobs)
elif ENCODING_SCHEME == "shared" and N_OBS == 1:
    HYPER_PARAMS = {'encoder_params': {'_nh': 50,
                                       '_nhlayers': 1,
                                       'alpha': 0.01,
                                       'early_stopping': True,
                                       'gamma': 0.001,
                                       'lamda': 0.001,
                                       'learning_rate': 0.001,
                                       'n_epochs': 50,
                                       'patience': 5},
                    'nn_params': {'input_shape': (17,), 'n_hidden_layers': 3, 'nh': 50}}


# hyperparams for all, n_obs =5, CVERR = 7.35% (2020-05-30), encoding_size=5 (+12 for knobs)
elif ENCODING_SCHEME == "all" and N_OBS == 5:
    HYPER_PARAMS = {'encoder_params': {'_nh': 50,
                                       '_nhlayers': 1,
                                       'alpha': 0.01,
                                       'early_stopping': True,
                                       'gamma': 0.001,
                                       'lamda': 0.001,
                                       'learning_rate': 0.001,
                                       'n_epochs': 50,
                                       'patience': 5},
                    'nn_params': {'input_shape': (17,), 'n_hidden_layers': 3, 'nh': 50}}


# hyperparams for all, n_obs = 1; CVERR = 10.46% (2020-05-30), encoding_size=5 (+12 for knobs)
elif ENCODING_SCHEME == "all" and N_OBS == 1:
    HYPER_PARAMS = {'encoder_params': {'_nh': 50,
                                       '_nhlayers': 1,
                                       'alpha': 0.01,
                                       'early_stopping': True,
                                       'gamma': 0.001,
                                       'lamda': 0.001,
                                       'learning_rate': 0.001,
                                       'n_epochs': 50,
                                       'patience': 5},
                    'nn_params': {'input_shape': (17,), 'n_hidden_layers': 3, 'nh': 50}}

DEBUG_MODE = False


def get_config_dict():
    dico = {
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
