import socket
hostname = socket.gethostname()

SEED = 10

N_WORKERS = 1  # w 5

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


LOGGING_LEVEL = 'INFO'

ENCODING_SIZE = 5

DATA_FNAME = "kpca-training-{}-{}--{}.npy".format(
    ENCODING_STRATEGY, N_OBS, hostname)


# Jan 20th, 2020
# When tuning the parameters of KPCA, we have fixed the n_components to 5...

# best params: shared, 5obs - cverr=8.48%
if ENCODING_STRATEGY == "shared" and N_OBS == 5:
    HYPER_PARAMS = {
        'kpca_params': {'gamma': 0.01, 'kernel': 'rbf', 'n_components': 5},
        'nn_params': {'input_shape': (15,),
                      'n_hidden_layers': 3, 'nh': 100}}

# best params: shared, 1obs - cverr=10%
elif ENCODING_STRATEGY == "shared" and N_OBS == 1:
    HYPER_PARAMS = {
        'kpca_params': {'gamma': None, 'kernel': 'rbf', 'n_components': 5},
        'nn_params': {'input_shape': (15,),
                      'n_hidden_layers': 4, 'nh': 50}}

# best params: all, 5obs - cverr=16.5%
elif ENCODING_STRATEGY == "all" and N_OBS == 5:
    HYPER_PARAMS = {
        'kpca_params': {'gamma': 0.001, 'kernel': 'rbf', 'n_components': 5},
        'nn_params': {'input_shape': (15,),
                      'n_hidden_layers': 4, 'nh': 100}}


# best params: all, 1obs - cverr=21.02%
elif ENCODING_STRATEGY == "all" and N_OBS == 1:
    HYPER_PARAMS = {
        'kpca_params': {'gamma': 10, 'kernel': 'rbf', 'n_components': 5},
        'nn_params': {'input_shape': (15,),
                      'n_hidden_layers': 3, 'nh': 20}}
else:
    raise NotImplementedError(
        "There are no best hyper-parameters yet for this option...")


def get_config_dict():
    dico = {
        'SEED': SEED,
        'LODS_FOLDER_PATH': LODS_FOLDER_PATH,
        'LODS_FNAME': LODS_FNAME,
        'ENCODING_SIZE': ENCODING_SIZE,
        'ENCODING_STRATEGY': ENCODING_STRATEGY,
        'N_OBS': N_OBS,
        'LOGGING_LEVEL': LOGGING_LEVEL,
        'DATA_FNAME': DATA_FNAME,
        'HYPER_PARAMS': HYPER_PARAMS,
        'N_TRAIN_PER_JOB': N_TRAIN_PER_JOB,
        'N_SHARED_TRAIN_PER_JOB': N_SHARED_TRAIN_PER_JOB
    }
    return dico
