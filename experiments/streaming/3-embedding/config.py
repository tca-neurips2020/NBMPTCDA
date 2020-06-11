OBSERVATION_SCHEME = "shared"
# OBSERVATION_SCHEME = "all"

N_TRAIN_PER_JOB = -1

N_SHARED_TRAIN_PER_JOB = -1

N_OBS = 5

ENCODING_SIZE = 5

N_RUNS = 10

N_WORKERS = 10

# tuning results collected on 2020-05-23; cverr=22.74%
if OBSERVATION_SCHEME == "all" and N_OBS == 5:
    PARAMS = {'activation': 'relu',
              'depth': 5,
              'dim_embedding': 10,
              'early_stopping': True,
              'n_epochs': 1000,
              'n_inc': 1000,
              'nh': 100,
              'patience': 50,
              'patience_inc': 50}

# tuning results collected on 2020-05-23; cverr=33.39%
elif OBSERVATION_SCHEME == "shared" and N_OBS == 5:
    PARAMS = {'activation': 'relu',
              'depth': 5,
              'dim_embedding': 5,
              'early_stopping': True,
              'n_epochs': 500,
              'n_inc': 500,
              'nh': 100,
              'patience': 50,
              'patience_inc': 20}
else:
    PARAMS = None

VOCAB_SIZE = 80  # maximum number of workloads that we have...

N_INC = PARAMS['n_inc']  # number of incremental epochs
# patience for early stopping while incrementally training
PATIENCE_INC = PARAMS['patience_inc']
ES_INC = True  # early stopping for incremental training

del PARAMS['n_inc']
del PARAMS['patience_inc']

LOGGING_LEVEL = "INFO"


# path to lods that should be read...
LODS_FOLDER_PATH = "../../../preprocessing/streaming/output/"
LODS_FNAME = "lods_mul.bin"


# name of the output file that will contain the data
DATA_FNAME = "training_embedding_OBS_SCHEME_{}_N_OBS_{}".format(
    OBSERVATION_SCHEME,
    N_OBS)


def get_config_dict():
    dico = {
        "OBSERVATION_SCHEME": OBSERVATION_SCHEME,
        "N_OBS": N_OBS,
        "ENCODING_SIZE": ENCODING_SIZE,
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
