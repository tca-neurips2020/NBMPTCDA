import socket
HOSTNAME = socket.gethostname()

N_TRAIN_PER_JOB = -1

N_SHARED_TRAIN_PER_JOB = -1


N_RUNS = 10

N_OBS = 1

LOGGING_LEVEL = 'DEBUG'

# OBSERVED_SCHEME = "shared"  # observing configurations that are shared
OBSERVED_SCHEME = "not-shared"

# REG = "RF"  # Random forest estimator
REG = "GPNP"  # gaussian process numpy implementation


############ OTTERTUNE CONFIGURATION PARAMETERS ######################
N_COMPONENTS = 5  # number of components to be used for factor analysis...


# ---GPR CONSTANTS---
DEFAULT_LENGTH_SCALE = 1.0

DEFAULT_MAGNITUDE = 1.0  # (KZ): also called signal variance...

DEFAULT_RIDGE = 0.01

#  Max training size in GPR model
MAX_TRAIN_SIZE = 7000

#  Batch size in GPR model
BATCH_SIZE = 3000

# Threads for TensorFlow config
NUM_THREADS = 4

# ---GRADIENT DESCENT CONSTANTS---
#  the maximum iterations of gradient descent
MAX_ITER = 500


DEFAULT_LEARNING_RATE = 0.01

DEFAULT_EPSILON = 1e-6

DEFAULT_SIGMA_MULTIPLIER = 3.0

DEFAULT_MU_MULTIPLIER = 1.0
######################################################################


################ OTHER PARAMS ########################################
SEED = 10  # Seed used to get reproducible results...

# path to config that generated the lods
LODS_FNAME = "lods_mul.bin"

# path to lods that should be read...
LODS_FOLDER_PATH = "../../../preprocessing/streaming/output/"

# Watch out this hard coded list used to report results on ML jobs...
ML_TEST_JOBS = [56, 60, 67, 70, 77, 79]


OUTPUT_FNAME = "ottertune_output_{}_nobs_{}_{}--{}.npy".format(
    OBSERVED_SCHEME, N_OBS, REG, HOSTNAME)


CONFIG = {
    'HOSTNAME': HOSTNAME,
    'N_COMPONENTS': N_COMPONENTS,
    'DEFAULT_LENGTH_SCALE': DEFAULT_LENGTH_SCALE,
    'DEFAULT_MAGNITUDE': DEFAULT_MAGNITUDE,
    'MAX_TRAIN_SIZE': MAX_TRAIN_SIZE,
    'BATCH_SIZE': BATCH_SIZE,
    'NUM_THREADS': NUM_THREADS,
    'MAX_ITER': MAX_ITER,
    'DEFAULT_RIDGE': DEFAULT_RIDGE,
    'DEFAULT_LEARNING_RATE': DEFAULT_LEARNING_RATE,
    'DEFAULT_EPSILON': DEFAULT_EPSILON,
    'DEFAULT_SIGMA_MULTIPLIER': DEFAULT_SIGMA_MULTIPLIER,
    'DEFAULT_MU_MULTIPLIER': DEFAULT_MU_MULTIPLIER,
    'SEED': SEED,
    'LODS_FNAME': LODS_FNAME,
    'LODS_FOLDER_PATH': LODS_FOLDER_PATH,
    'ML_TEST_JOBS': ML_TEST_JOBS,
    'N_OBS': N_OBS,
    'LOGGING_LEVEL': LOGGING_LEVEL,
    'OBSERVED_SCHEME': OBSERVED_SCHEME,
    'OUTPUT_FNAME': OUTPUT_FNAME,
    'N_RUNS': N_RUNS,
    'REG': REG,
    'N_TRAIN_PER_JOB': N_TRAIN_PER_JOB,
    'N_SHARED_TRAIN_PER_JOB': N_SHARED_TRAIN_PER_JOB
}
