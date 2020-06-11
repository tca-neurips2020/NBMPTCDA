import os
import numpy as np
import tensorflow as tf
from time import time
from trainingutils import translate_X_y
from sparkmodeling.common.lodatastruct import LODataStruct
from sparkmodeling.autoencoder.contractiveplus import CAEPlus
from sparkmodeling.nn.nnregressor import NNregressor
from trainingutils import persist_data
from trainingutils import extract_and_set_encoding
import logging
from config import *
import time
import socket
from trainingutils import extract_encoding_and_map_to_nearest
from multiprocessing import Manager, Queue, Process
from trainingutils import copyDict
from trainingutils import print_err_aggregates
from sklearn.utils import shuffle


def init_distributed_dicts(lods):
    manager = Manager()
    err_types = ['no-opt', 'map']

    test_aliases = sorted(list(set(lods.test.a.ravel())))
    test_jobs = list(map(lambda x: lods.alias_to_id[x], test_aliases))

    errs = manager.dict()
    for err_type in err_types:
        errs[err_type] = manager.dict()
        for test_job in test_jobs:
            errs[err_type][test_job] = manager.list()

    data = manager.dict()
    data['errs'] = errs
    data['autoencoders'] = manager.list()
    data['regressors'] = manager.list()
    data['training_errs'] = manager.list()

    data['mappings'] = manager.dict()
    for test_job in test_jobs:
        data['mappings'][test_job] = manager.list()

    return data


class OneWorkerProcess(Process):
    def __init__(self, lods, q, out_data):
        Process.__init__(self)
        self.lods = lods
        self.q = q
        self.out_data = out_data

    def run(self):
        lods = self.lods
        while True:
            seed = self.q.get()
            if seed == -1:
                break
            train_model_and_evaluate(
                lods, self.out_data, seed=seed)


def evaluate(test_job, cae,
             reg, lods, observed_traces,
             observed_traces_slices, out_data, test_aliases=[]):
    # Get the trace(s) for this specific test job
    idx = observed_traces_slices[test_job]
    trace = observed_traces.a[idx], observed_traces.X[idx], \
        observed_traces.Y[idx]

    # Extract encoding from given trace(s) and trained encoder
    extract_and_set_encoding(cae, trace)
    proxy = extract_encoding_and_map_to_nearest(
        cae, trace, lods.alias_to_id, test_aliases)
    out_data['mappings'][test_job].append(proxy)

    # Test data (without encodings)
    test = lods.test
    slices_test = test.slice_by_job_id(lods.alias_to_id)
    idxs_test = slices_test[test_job]
    X_test = np.hstack([test.a, test.X])[idxs_test, :]
    y_test = test.targets.ravel()[idxs_test]

    # Test data for regressor prediction (without mapping)
    X_test_, y_test_ = translate_X_y(
        X_test, y_test, cae.centroids)

    # Test data for regressor prediction (with mapping)
    X_test__, y_test__ = translate_X_y(
        X_test, y_test, cae.altered_centroids)

    reg_ = reg.clone()
    err = reg_.MAPE(X_test_, y_test_)
    logging.info(
        "[Test job: {}] \t Error no-opt: {:.2f}".format(test_job, err))

    reg_ = reg.clone()
    assert np.abs(err - reg_.MAPE(X_test_, y_test_)) < 1e-10
    # Now let's use the data from mapping...
    err_mapping = reg_.MAPE(X_test__, y_test__)
    logging.info("[Test job: {}] \t Error map: {:.2f}".format(test_job,
                                                              err_mapping))

    # err, err_mapping
    out_data['errs']['no-opt'][test_job].append(err)
    out_data['errs']['map'][test_job].append(err_mapping)


def trim_lods(lods):
    """ Trim LODS according to # of shared and # of non-shared
    configurations given in the input configuration file...
    """
    if N_TRAIN_PER_JOB != -1:
        lods.trainval = lods.trainval.get_x(N_TRAIN_PER_JOB)
        logging.info("shape of remaining trainval (X): {}".format(
            lods.trainval.X.shape))
    if N_SHARED_TRAIN_PER_JOB != -1:
        lods.shared_trainval = lods.shared_trainval.get_x(
            N_SHARED_TRAIN_PER_JOB)
        logging.info(
            "shape of remaining shared trainval (X): {}".format(
                lods.shared_trainval.X.shape))


def compute_centroids(encoder, lods, scheme='shared'):
    """computes centroids for training data"""
    centroids = {}
    if scheme == 'shared':
        ds = lods.shared_trainval.get_x(N_OBS)
    else:
        ds = lods.shared_trainval + lods.trainval
    slices = ds.slice_by_aliases()
    for a in slices:
        idxs = slices[a]
        encodings = encoder.transform(ds.Y[idxs, :])
        centroids[a] = np.mean(encodings, axis=0)

    encoder.centroids = centroids


def train_model_and_evaluate(lods, out_data, seed=10):
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    ds_train = lods.trainval + lods.shared_trainval

    # 2. train the autoencoder
    ae_params = HYPER_PARAMS['ae_params']
    ae_params['input_dim'] = ds_train.Y.shape[1]  # 561 metrics

    cae = CAEPlus.build(**ae_params)
    cae.compile()
    cae.fit(ds_train.Y, ds_train.X, log_time=True)

    # 3. extract encodings for training workloads
    compute_centroids(cae, lods, scheme=ENCODING_SCHEME)

    # 4. fetch training data for the regressor
    X_ = np.hstack(
        [ds_train.a, ds_train.X])
    y_ = ds_train.targets.ravel()
    X, y = translate_X_y(X_, y_, cae.centroids)

    # 5. Train a regressor on the training data
    nn_params = HYPER_PARAMS['nn_params']
    logging.info(
        "Fitting regressor on data of shapes: {}, {}".format(
            X.shape, y.shape))
    reg = NNregressor(**nn_params, v1_compat_mode=True,
                      keras_2=True, random_state=seed)
    reg.fit(X, y, log_time=True)
    training_mape = reg.MAPE(X, y)
    # 6. calculate the training error
    logging.info("Training Error: {:.2f}%".format(training_mape))
    out_data['training_errs'].append(training_mape)

    # 7. get observed traces and evaluate on test jobs...
    if ENCODING_SCHEME == 'shared':
        observed_traces = lods.shared_traincomplement.get_x(N_OBS)
    else:
        observed_traces = lods.traincomplement.get_x(N_OBS)

    logging.info("observed_traces description: ")
    observed_traces.describe()

    observed_traces_slices = observed_traces.slice_by_job_id(
        alias_to_id=lods.alias_to_id)

    test_aliases = sorted(list(set(lods.test.a.ravel())))

    for test_job in observed_traces_slices:
        evaluate(test_job, cae, reg, lods, observed_traces,
                 observed_traces_slices, out_data, test_aliases)

    # Append trained encoder information (with centroids) to output_data
    out_data['autoencoders'].append(cae.get_persist_info())

    # Append trained regressor information to output_data
    out_data['regressors'].append(reg.get_persist_info())

    persist_data(copyDict(out_data), DATA_FNAME)


def get_lods(describe=False):
    # autobuild is set to false because persisted as built object.
    lods = LODataStruct.load_from_file(os.path.join(
        LODS_FOLDER_PATH, LODS_FNAME), autobuild=False)

    # Overwrite folder containing csvs and autobuild
    lods.id_to_fname = None  # backward compatibility for streaming
    csv_folder = "../../../../datasets/streaming/"
    lods.folder = csv_folder
    lods._autobuild()

    lods.minmaxscale("X")
    lods.minmaxscale("Y")
    if describe:
        logging.info("BEFORE TRIMMING: ")
        lods.describe()
    trim_lods(lods)
    if describe:
        logging.info("AFTER TRIMMING: ")
        lods.describe()

    return lods


def main():
    t0 = time.time()
    np.random.seed(SEED)
    logging.basicConfig(level=getattr(logging, LOGGING_LEVEL))
    lods = get_lods(describe=True)

    lods_copy = get_lods()
    lods_copy.destroy_data()

    out_data = init_distributed_dicts(lods)
    out_data['lods'] = lods_copy
    out_data['CONFIG'] = get_config_dict()

    q = Queue()
    for i in range(N_RUNS):
        q.put(i)
    for i in range(N_WORKERS):
        q.put(-1)  # -1 as a flag for stopping

    processes = []
    for i in range(N_WORKERS):
        p = OneWorkerProcess(lods, q, out_data)
        p.start()
        os.system("taskset -p -c %d %d" % ((i % os.cpu_count()), p.pid))
        processes.append(p)

    for p in processes:
        p.join()

    print_err_aggregates(out_data['errs'])
    t_final = time.time()
    script_duration = t_final - t0
    logging.info("Script total running time: {} minutes and {} seconds".format(
        script_duration // 60, int(script_duration % 60)))


if __name__ == '__main__':
    main()
