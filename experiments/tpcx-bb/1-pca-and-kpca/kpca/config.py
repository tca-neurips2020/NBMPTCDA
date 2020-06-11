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
LODS_FOLDER_PATH = "../../../../preprocessing/tpcx-bb/output/"


# centroids_strategy: shared or all
ENCODING_STRATEGY = "shared"  # (with split intersections)
# ENCODING_STRATEGY = "all"


N_OBS = 1


LOGGING_LEVEL = 'INFO'

ENCODING_SIZE = 10


DATA_FNAME = "kpca-training-{}-{}--{}.npy".format(
    ENCODING_STRATEGY, N_OBS, hostname)

# 2020-06-05
# When tuning the parameters of KPCA, we have fixed the n_components to 10...
# best params: shared, 5obs - cverr=15.98%; encoding dimension=5
if ENCODING_STRATEGY == "shared" and N_OBS == 5:
    HYPER_PARAMS = {
        'kpca_params': {'gamma': 0.01, 'kernel': 'rbf', 'n_components': 10},
        'nn_params': {'input_shape': (22,),
                      'n_hidden_layers': 3, 'nh': 50}}

# 2020-06-05
# best params: shared, 1obs - cverr=19.99%; encoding dimension=5
elif ENCODING_STRATEGY == "shared" and N_OBS == 1:
    HYPER_PARAMS = {
        'kpca_params': {'gamma': 0.01, 'kernel': 'rbf', 'n_components': 10},
        'nn_params': {'input_shape': (22,),
                      'n_hidden_layers': 3, 'nh': 50}}

# 2020-06-05
# best params: all, 5obs - cverr=37.09%; encoding dimension=5
elif ENCODING_STRATEGY == "all" and N_OBS == 5:
    HYPER_PARAMS = {
        'kpca_params': {'gamma': 0.01, 'kernel': 'rbf', 'n_components': 10},
        'nn_params': {'input_shape': (22,),
                      'n_hidden_layers': 2, 'nh': 5}}

# 2020-05-31
# best params: all, 1obs - cverr=60.27%; encoding dimension=5
elif ENCODING_STRATEGY == "all" and N_OBS == 1:
    HYPER_PARAMS = {
        'kpca_params': {'gamma': 100, 'kernel': 'rbf', 'n_components': 5},
        'nn_params': {'input_shape': (17,),
                      'n_hidden_layers': 4, 'nh': 10}}


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
