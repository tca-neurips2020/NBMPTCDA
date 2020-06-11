import socket
hostname = socket.gethostname()

SEED = 10

N_WORKERS = 5

N_RUNS = 10

N_TRAIN_PER_JOB = -1

N_SHARED_TRAIN_PER_JOB = -1


LODS_FNAME = "lods_mul.bin"

# path to lods that should be read...
LODS_FOLDER_PATH = "../../../../preprocessing/streaming/output/"


# centroids_strategy: shared or all
ENCODING_STRATEGY = "shared"
# ENCODING_STRATEGY = "all"

ML_TEST_JOBS = [56, 60, 67, 70, 77, 79]

N_OBS = 5

ENCODING_SIZE = 5


LOGGING_LEVEL = 'INFO'


DATA_FNAME = "pca-training-{}-{}--{}.npy".format(
    ENCODING_STRATEGY, N_OBS, hostname)

# Jan 20th, 2020
# best params: shared, 5_obs, cverr=9.32%
if ENCODING_STRATEGY == "shared" and N_OBS == 5:
    HYPER_PARAMS = {
        'nn_params': {'input_shape': (15,), 'n_hidden_layers': 3, 'nh': 100}}

# # best params: shared, 1 obs, cverr=9.85%
elif ENCODING_STRATEGY == "shared" and N_OBS == 1:
    HYPER_PARAMS = {
        'nn_params': {'input_shape': (15,), 'n_hidden_layers': 4, 'nh': 100}}

# # best params: all, 5 obs, cverr=20.95%
elif ENCODING_STRATEGY == "all" and N_OBS == 5:
    HYPER_PARAMS = {
        'nn_params': {'input_shape': (15,), 'n_hidden_layers': 4, 'nh': 10}}


# # best params: all, 1 obs, cverr=41.93%
elif ENCODING_STRATEGY == "all" and N_OBS == 1:
    HYPER_PARAMS = {
        'nn_params': {'input_shape': (15,), 'n_hidden_layers': 2, 'nh': 10}}


def get_config_dict():
    dico = {
        'SEED': SEED,
        'LODS_FOLDER_PATH': LODS_FOLDER_PATH,
        'LODS_FNAME': LODS_FNAME,
        'ENCODING_STRATEGY': ENCODING_STRATEGY,
        'N_OBS': N_OBS,
        'ENCODING_SIZE': ENCODING_SIZE,
        'LOGGING_LEVEL': LOGGING_LEVEL,
        'DATA_FNAME': DATA_FNAME,
        'HYPER_PARAMS': HYPER_PARAMS,
        'N_TRAIN_PER_JOB': N_TRAIN_PER_JOB,
        'N_SHARED_TRAIN_PER_JOB': N_SHARED_TRAIN_PER_JOB
    }
    return dico
