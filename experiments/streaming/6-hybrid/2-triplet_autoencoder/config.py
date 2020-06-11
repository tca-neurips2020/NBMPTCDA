import socket
hostname = socket.gethostname()

SEED = 10

N_WORKERS = 4

N_RUNS = 8

N_TRAIN_PER_JOB = -1

N_SHARED_TRAIN_PER_JOB = -1

assert N_SHARED_TRAIN_PER_JOB != 0

assert N_TRAIN_PER_JOB != 0

# path to config that generated the lods
LODS_FNAME = "lods_mul.bin"

# path to lods that should be read...
LODS_FOLDER_PATH = "../../../../preprocessing/streaming/output/"


# Observation Scheme: shared or all
ENCODING_SCHEME = "shared"  #
# ENCODING_SCHEME = "all"


ML_TEST_JOBS = [56, 60, 67, 70, 77, 79]


N_OBS = 5

ENCODING_SIZE = 5


LOGGING_LEVEL = 'INFO'

DATA_FNAME = "training-data-and-stats_{}_nobs_{}--{}.npy".format(
    ENCODING_SCHEME,
    N_OBS,
    hostname)


HYPER_PARAMS = {
    'autoencoder_params':
    {'layer_sizes': [20, ENCODING_SIZE],
     'activations': ['relu', None],
     'alpha': 0.1, 'early_stopping': True, 'learning_rate': 0.01,
     'gamma': 0.001,
     'n_epochs': 50, 'patience': 5},
    'nn_params': {'input_shape': (15,),
                  'n_hidden_layers': 3, 'nh': 50}}


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
