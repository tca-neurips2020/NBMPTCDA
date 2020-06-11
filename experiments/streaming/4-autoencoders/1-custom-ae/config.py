import socket
hostname = socket.gethostname()

SEED = 10

N_WORKERS = 5

N_RUNS = 10

N_TRAIN_PER_JOB = -1

N_SHARED_TRAIN_PER_JOB = -1

# path to config that generated the lods
LODS_FNAME = "lods_mul.bin"

# path to lods that should be read...
LODS_FOLDER_PATH = "../../../../preprocessing/streaming/output/"


# centroids_strategy: shared or all
ENCODING_STRATEGY = "shared"  # (with split intersections)
# ENCODING_STRATEGY = "all"


ML_TEST_JOBS = [56, 60, 67, 70, 77, 79]

N_OBS = 5

# This should correspond to the value that was used when tuning the parameters
ENCODING_SIZE = 5


# Fix the value of lamda:
LAMDA = 1e-2


LOGGING_LEVEL = 'INFO'

DATA_FNAME = "training-data-and-stats_{}_nobs_{}--{}.npy".format(
    ENCODING_STRATEGY,
    N_OBS,
    hostname)

# Jan 19th, 2019
# hyperparams for shared, n_obs = 5; CVERR=11.40%
if ENCODING_STRATEGY == "shared" and N_OBS == 5:
    HYPER_PARAMS = {'ae_params': {'activation': 'relu',
                                  'depth': 2,
                                  'early_stopping': True,
                                  'encoding_dim': 5,
                                  'initial_learning_rate': 0.0001,
                                  'lamda': 0.01,
                                  'max_refit_attempts': 3,
                                  'n_iter': 1000,
                                  'nh': 20,
                                  'patience': 100},
                    'nn_params': {'input_shape': (15,), 'n_hidden_layers': 3, 'nh': 50}}


# hyperparams for shared, n_obs = 1; CVERR = 11.74%
elif ENCODING_STRATEGY == "shared" and N_OBS == 1:
    HYPER_PARAMS = {'ae_params': {'activation': 'relu',
                                  'depth': 1,
                                  'early_stopping': True,
                                  'encoding_dim': 5,
                                  'initial_learning_rate': 0.0001,
                                  'lamda': 0.01,
                                  'max_refit_attempts': 3,
                                  'n_iter': 1000,
                                  'nh': 10,
                                  'patience': 100},
                    'nn_params': {'input_shape': (15,), 'n_hidden_layers': 3, 'nh': 100}}


# hyperparams for all, n_obs=5 (cverr=13.37%)
elif ENCODING_STRATEGY == "all" and N_OBS == 5:
    HYPER_PARAMS = {'ae_params': {'activation': 'relu',
                                  'depth': 1,
                                  'early_stopping': True,
                                  'encoding_dim': 5,
                                  'initial_learning_rate': 0.0001,
                                  'lamda': 0.01,
                                  'max_refit_attempts': 3,
                                  'n_iter': 1000,
                                  'nh': 5,
                                  'patience': 100},
                    'nn_params': {'input_shape': (15,), 'n_hidden_layers': 4, 'nh': 50}}


# hyperparams for all, n_obs=1 (cverr=23.56%)
elif ENCODING_STRATEGY == "all" and N_OBS == 1:
    HYPER_PARAMS = {'ae_params': {'activation': 'relu',
                                  'depth': 3,
                                  'early_stopping': True,
                                  'encoding_dim': 5,
                                  'initial_learning_rate': 0.1,
                                  'lamda': 0.01,
                                  'max_refit_attempts': 3,
                                  'n_iter': 1000,
                                  'nh': 50,
                                  'patience': 100},
                    'nn_params': {'input_shape': (15,), 'n_hidden_layers': 2, 'nh': 100}}


def get_config_dict():
    dico = {
        'SEED': SEED,
        'LODS_FOLDER_PATH': LODS_FOLDER_PATH,
        'LODS_FNAME': LODS_FNAME,
        'ENCODING_STRATEGY': ENCODING_STRATEGY,
        'N_OBS': N_OBS,
        'ENCODING_SIZE': ENCODING_SIZE,
        'LAMDA': LAMDA,
        'LOGGING_LEVEL': LOGGING_LEVEL,
        'DATA_FNAME': DATA_FNAME,
        'HYPER_PARAMS': HYPER_PARAMS,
        'N_TRAIN_PER_JOB': N_TRAIN_PER_JOB,
        'N_SHARED_TRAIN_PER_JOB': N_SHARED_TRAIN_PER_JOB

    }
    return dico
