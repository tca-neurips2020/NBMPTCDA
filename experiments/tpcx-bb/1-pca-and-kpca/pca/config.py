import socket
hostname = socket.gethostname()

SEED = 10

N_WORKERS = 10

N_RUNS = 10

N_TRAIN_PER_JOB = -1

N_SHARED_TRAIN_PER_JOB = -1


# path to config that generated the lods
LODS_FNAME = "lods_mul.bin"

# path to lods that should be read...
LODS_FOLDER_PATH = "../../../../preprocessing/tpcx-bb/output/"


# centroids_strategy: shared or all
ENCODING_STRATEGY = "shared"
# ENCODING_STRATEGY = "all"

N_OBS = 5

ENCODING_SIZE = 10


LOGGING_LEVEL = 'INFO'


DATA_FNAME = "pca-training-{}-{}--{}.npy".format(
    ENCODING_STRATEGY, N_OBS, hostname)

# 2020-06-05
# best params: shared, 5_obs, cverr=13.30%; encoding size=10
if ENCODING_STRATEGY == "shared" and N_OBS == 5:
    HYPER_PARAMS = {'nn_params': {'input_shape': (
        22,), 'n_hidden_layers': 4, 'nh': 50}}

# 2020-06-05
# # best params: shared, 1 obs, cverr=19.55%; encoding size=10
elif ENCODING_STRATEGY == "shared" and N_OBS == 1:
    HYPER_PARAMS = {'nn_params': {'input_shape': (
        22,), 'n_hidden_layers': 3, 'nh': 100}}

# 2020-06-05
# # best params: all, 5 obs, cverr=37.94%; encoding size=10
elif ENCODING_STRATEGY == "all" and N_OBS == 5:
    HYPER_PARAMS = {'nn_params': {'input_shape': (
        22,), 'n_hidden_layers': 4, 'nh': 50}}

# 2020-06-05
# # best params: all, 1 obs, cverr=52.22%; encoding size=10
elif ENCODING_STRATEGY == "all" and N_OBS == 1:
    HYPER_PARAMS = {'nn_params': {'input_shape': (
        22,), 'n_hidden_layers': 1, 'nh': 5}}


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
