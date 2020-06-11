import socket
hostname = socket.gethostname()

SEED = 10

N_WORKERS = 10

N_RUNS = 10

N_TRAIN_PER_JOB = -1

N_SHARED_TRAIN_PER_JOB = -1

# path to config that generated the lods
LODS_FNAME = "lods_mul.bin"


LODS_FOLDER_PATH = "../../../../preprocessing/tpcx-bb/output/"


ENCODING_STRATEGY = "shared"  # (with split intersections)
# ENCODING_STRATEGY = "all"

N_OBS = 1

LOGGING_LEVEL = 'INFO'

DATA_FNAME = "training-data-and-stats_{}_nobs_{}--{}.npy".format(
    ENCODING_STRATEGY,
    N_OBS,
    hostname)

# 2020-06-01
# hyperparams for shared, n_obs = 5; CVERR=17.04%
if ENCODING_STRATEGY == "shared" and N_OBS == 5:
    HYPER_PARAMS = {'ae_params': {'activation': 'relu',
                                  'depth': 3,
                                  'early_stopping': True,
                                  'encoding_dim': 5,
                                  'initial_learning_rate': 0.001,
                                  'lamda': 1,
                                  'n_iter': 1000,
                                  'nh': 100,
                                  'patience': 100},
                    'nn_params': {'input_shape': (17,), 'n_hidden_layers': 2, 'nh': 50}}

# 2020-06-01
# hyperparams for shared, n_obs = 1; CVERR=22.85%
elif ENCODING_STRATEGY == "shared" and N_OBS == 1:
    HYPER_PARAMS = {'ae_params': {'activation': 'relu',
                                  'depth': 3,
                                  'early_stopping': True,
                                  'encoding_dim': 5,
                                  'initial_learning_rate': 0.001,
                                  'lamda': 1,
                                  'n_iter': 1000,
                                  'nh': 100,
                                  'patience': 100},
                    'nn_params': {'input_shape': (17,), 'n_hidden_layers': 2, 'nh': 50}}

# 2020-06-01
# hyperparams for all, n_obs=5 (cverr=14.43%) [!not with bs]
elif ENCODING_STRATEGY == "all" and N_OBS == 5:
    HYPER_PARAMS = {'ae_params': {'activation': 'relu',
                                  'depth': 2,
                                  'early_stopping': True,
                                  'encoding_dim': 10,
                                  'initial_learning_rate': 0.001,
                                  'lamda': 0.1,
                                  'n_iter': 1000,
                                  'nh': 100,
                                  'patience': 100},
                    'nn_params': {'input_shape': (22,), 'n_hidden_layers': 4, 'nh': 50}}


# hyperparams for all, n_obs=1 (cverr=30.96%) [!not with bs]
# 2020-06-01
elif ENCODING_STRATEGY == "all" and N_OBS == 1:
    HYPER_PARAMS = {'ae_params': {'activation': 'relu',
                                  'depth': 2,
                                  'early_stopping': True,
                                  'encoding_dim': 10,
                                  'initial_learning_rate': 0.001,
                                  'lamda': 0.1,
                                  'n_iter': 1000,
                                  'nh': 100,
                                  'patience': 100},
                    'nn_params': {'input_shape': (22,), 'n_hidden_layers': 4, 'nh': 50}}


def get_config_dict():
    dico = {
        'SEED': SEED,
        'LODS_FOLDER_PATH': LODS_FOLDER_PATH,
        'LODS_FNAME': LODS_FNAME,
        'ENCODING_STRATEGY': ENCODING_STRATEGY,
        'N_OBS': N_OBS,
        'LOGGING_LEVEL': LOGGING_LEVEL,
        'DATA_FNAME': DATA_FNAME,
        'HYPER_PARAMS': HYPER_PARAMS,
        'N_TRAIN_PER_JOB': N_TRAIN_PER_JOB,
        'N_SHARED_TRAIN_PER_JOB': N_SHARED_TRAIN_PER_JOB

    }
    return dico
