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

# path to lods that should be read...
LODS_FOLDER_PATH = "../../../../preprocessing/streaming/output/"


ENCODING_SCHEME = "shared"  #
# ENCODING_SCHEME = "all"


ML_TEST_JOBS = [56, 60, 67, 70, 77, 79]


N_OBS = 1

ENCODING_SIZE = 15


LOGGING_LEVEL = 'INFO'

DATA_FNAME = "training-data-and-stats_{}_nobs_{}--{}.npy".format(
    ENCODING_SCHEME,
    N_OBS,
    hostname)


# hyperparams for shared, n_obs=5 (cverr= 7.97%) (encoding_dim=15)
# cverr results from 2020-05-06
if ENCODING_SCHEME == "shared" and N_OBS == 5:
    HYPER_PARAMS = {'ae_params': {'beta': 2,
                                  'depth': 1,
                                  'early_stopping': True,
                                  'encoding_dim': 15,
                                  'intermediate_activation': 'relu',
                                  'learning_rate': 0.001,
                                  'n_epochs': 500,
                                  'nh': 100,
                                  'patience': 50,
                                  'recons_type': 'xent'},
                    'nn_params': {'input_shape': (25,), 'n_hidden_layers': 4, 'nh': 100}}


elif ENCODING_SCHEME == "shared" and N_OBS == 1:
    HYPER_PARAMS = {'ae_params': {'beta': 2,
                                  'depth': 1,
                                  'early_stopping': True,
                                  'encoding_dim': 15,
                                  'intermediate_activation': 'relu',
                                  'learning_rate': 0.001,
                                  'n_epochs': 500,
                                  'nh': 100,
                                  'patience': 50,
                                  'recons_type': 'xent'},
                    'nn_params': {'input_shape': (25,), 'n_hidden_layers': 4, 'nh': 100}}


# hyperparams for all, n_obs=5 (cverr= 15.41%) (encoding_dim=15)
# cverr results from 2020-05-04
elif ENCODING_SCHEME == "all" and N_OBS == 5:
    HYPER_PARAMS = {'ae_params': {'beta': 1.5,
                                  'depth': 2,
                                  'early_stopping': True,
                                  'encoding_dim': 15,
                                  'intermediate_activation': 'relu',
                                  'learning_rate': 0.001,
                                  'n_epochs': 500,
                                  'nh': 50,
                                  'patience': 50,
                                  'recons_type': 'xent'},
                    'nn_params': {'input_shape': (25,), 'n_hidden_layers': 4, 'nh': 100}}


# hyperparams for all, n_obs=1 (cverr= 20.75%) (encoding_dim=15)
# cverr results from 2020-05-04
elif ENCODING_SCHEME == "all" and N_OBS == 1:
    HYPER_PARAMS = {'ae_params': {'beta': 10,
                                  'depth': 2,
                                  'early_stopping': True,
                                  'encoding_dim': 15,
                                  'intermediate_activation': 'relu',
                                  'learning_rate': 0.01,
                                  'n_epochs': 500,
                                  'nh': 20,
                                  'patience': 50,
                                  'recons_type': 'xent'},
                    'nn_params': {'input_shape': (25,), 'n_hidden_layers': 3, 'nh': 50}}


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
