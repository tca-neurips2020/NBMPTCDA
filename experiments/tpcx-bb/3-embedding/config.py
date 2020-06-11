# OBSERVATION_SCHEME = "shared"
OBSERVATION_SCHEME = "all"

N_TRAIN_PER_JOB = -1

N_SHARED_TRAIN_PER_JOB = -1

N_OBS = 5  # Doesn't make sense to be less than encoding size...

N_RUNS = 10

N_WORKERS = 10

# tuning results collected on 2020-05-31; cverr=12.56%; best encoding size found: 10
# 10_relu_5_100_500_50_True_100_50
if OBSERVATION_SCHEME == "all" and N_OBS == 5:
    PARAMS = {'activation': 'relu',
              'depth': 5,
              'dim_embedding': 10,
              'early_stopping': True,
              'n_epochs': 500,
              'n_inc': 100,
              'nh': 100,
              'patience': 50,
              'patience_inc': 50}

# tuning results collected on 2020-05-31; cverr=13.53%; best encoding size found:10
# 10_relu_5_100_500_50_True_100_50
elif OBSERVATION_SCHEME == "shared" and N_OBS == 5:
    PARAMS = {'activation': 'relu',
              'depth': 5,
              'dim_embedding': 10,
              'early_stopping': True,
              'n_epochs': 500,
              'n_inc': 100,
              'nh': 100,
              'patience': 50,
              'patience_inc': 50}
else:
    PARAMS = None

# N_INC = 1000  # number of incremental epochs
# PATIENCE_INC = 25  # patience for early stopping while incrementally training
# ES_INC = True  # early stopping for incremental training


VOCAB_SIZE = 1160  # maximum number of workloads that we have...

N_INC = PARAMS['n_inc']  # number of incremental epochs
# patience for early stopping while incrementally training
PATIENCE_INC = PARAMS['patience_inc']
ES_INC = True  # early stopping for incremental training

del PARAMS['n_inc']
del PARAMS['patience_inc']

LOGGING_LEVEL = "INFO"


LODS_FOLDER_PATH = "../../../preprocessing/tpcx-bb/output/"
LODS_FNAME = "lods_mul.bin"


# name of the output file that will contain the data
DATA_FNAME = "training_embedding_OBS_SCHEME_{}_N_OBS_{}".format(
    OBSERVATION_SCHEME,
    N_OBS)


def get_config_dict():
    dico = {
        "OBSERVATION_SCHEME": OBSERVATION_SCHEME,
        "N_OBS": N_OBS,
        "N_RUNS": N_RUNS,
        "N_WORKERS": N_WORKERS,
        "PARAMS": PARAMS,
        "VOCAB_SIZE": VOCAB_SIZE,
        "N_INC": N_INC,
        "PATIENCE_INC": PATIENCE_INC,
        "ES_INC": ES_INC,
        "LOGGING_LEVEL": LOGGING_LEVEL,
        "N_TRAIN_PER_JOB": N_TRAIN_PER_JOB,
        "N_SHARED_TRAIN_PER_JOB": N_SHARED_TRAIN_PER_JOB,
        "LODS_FOLDER_PATH": LODS_FOLDER_PATH,
        "LODS_FNAME": LODS_FNAME,
        "DATA_FNAME": DATA_FNAME
    }
