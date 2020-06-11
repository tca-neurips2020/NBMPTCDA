import numpy as np

DATA_FOLDER = "../../datasets/streaming"

CONFIG_PATH = "config.json"

OUTPUT_FOLDER = "output--"

WITH_INTENSIVE = False

SEPARATE_INTERSECTIONS_MUL = True

SEPARATE_INTERSECTIONS_X = True

SHARED_WITHIN_TEMPLATES = False

INTENSIVE_WORKLOADS = [1, 2, 3, 4, 7, 8]
PARAM_WORKLOADS = list(np.arange(10, 80))  # 79 is the id of the last workload
TEMPLATES = {
    "A": list(np.arange(10, 22)),
    "B": list(np.arange(22, 26)),
    "C": list(np.arange(26, 32)),
    "D": list(np.arange(32, 44)),
    "E": list(np.arange(44, 56)),
    "F": list(np.arange(56, 68)),
    "G": list(np.arange(68, 80)),
}

TEST_WORKLOADS = {
    "A": [10, 15, 20],
    "B": [22],
    "C": [30],
    "D": [33, 38, 43],
    "E": [46, 50, 55],
    "F": [56, 60, 67],
    "G": [70, 77, 79]
}


DESTROY_ON_SERIALIZE = False


def get_config_dict():
    dico = {
        'DATA_FOLDER': DATA_FOLDER,
        'CONFIG_PATH': CONFIG_PATH,
        'OUTPUT_FOLDER': OUTPUT_FOLDER,
        'WITH_INTENSIVE': WITH_INTENSIVE,
        'SEPARATE_INTERSECTIONS_MUL': SEPARATE_INTERSECTIONS_MUL,
        'SEPARATE_INTERSECTIONS_X': SEPARATE_INTERSECTIONS_X,
        'INTENSIVE_WORKLOADS': INTENSIVE_WORKLOADS,
        'PARAM_WORKLOADS': PARAM_WORKLOADS,
        'TEMPLATES': TEMPLATES,
        'TEST_WORKLOADS': TEST_WORKLOADS,
        'DESTROY_ON_SERIALIZE': DESTROY_ON_SERIALIZE,
        'SHARED_WITHIN_TEMPLATES': SHARED_WITHIN_TEMPLATES
    }
    return dico
