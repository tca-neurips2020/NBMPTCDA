import os
import numpy as np
import tensorflow as tf
from time import time
from trainingutils import translate_X_y
from sparkmodeling.common.lodatastruct import LODataStruct
from sparkmodeling.nn.nnregressor import NNregressor
from trainingutils import persist_data
from trainingutils import extract_and_set_encoding
from sklearn.decomposition import KernelPCA
import logging
from config import *
import time
import socket
from trainingutils import extract_encoding_and_map_to_nearest
from multiprocessing import Manager, Queue, Process
from trainingutils import copyDict
from trainingutils import print_err_aggregates
from trainingutils import compute_centroids
from trainingutils import extract_encoding


def init_distributed_dicts(lods):
    manager = Manager()
    err_types = ['no-opt', 'cal', 'map', 'map_then_cal']

    test_aliases = sorted(list(set(lods.test.a.ravel())))
    test_jobs = list(map(lambda x: lods.alias_to_id[x], test_aliases))

    errs = manager.dict()
    for err_type in err_types:
        errs[err_type] = manager.dict()
        for test_job in test_jobs:
            errs[err_type][test_job] = manager.list()

    data = manager.dict()
    data['errs'] = errs
    data['kpcas'] = manager.list()
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
            train_model_and_evaluate(lods, self.out_data, seed=seed)


def evaluate(test_job, pca,
             reg, lods, observed_traces,
             observed_traces_slices, out_data, test_aliases=[]):
    n_knob_cols = len(lods.config['COLS_KNOBS'])

    # Get the trace(s) for this specific test job
    idx = observed_traces_slices[test_job]
    trace = observed_traces.a[idx], observed_traces.X[idx], \
        observed_traces.Y[idx]

    # Extract encoding from given trace(s) and trained pca
    a_new, missing_centroid = extract_encoding(pca, trace,
                                               use_full_trace=False)
    pca.centroids[a_new] = missing_centroid

    proxy = extract_encoding_and_map_to_nearest(
        pca, trace, lods.alias_to_id, test_aliases,
        within_template=False, metric='euclidean')
    out_data['mappings'][test_job].append(proxy)

    # Calibration data
    X_calib = np.hstack(
        [observed_traces.a[idx],
            observed_traces.X[idx]])
    y_calib = observed_traces.targets.ravel()[idx]

    # Calibration Data for regressor ('without mapping' case):
    X_calib_, y_calib_ = translate_X_y(
        X_calib, y_calib, pca.centroids, n_knob_cols)
    # Calibration Data for regressor ('with mapping' case):
    X_calib__, y_calib__ = translate_X_y(
        X_calib, y_calib, pca.altered_centroids, n_knob_cols)

    # Test data (without encodings)
    test = lods.test
    slices_test = test.slice_by_job_id(lods.alias_to_id)
    idxs_test = slices_test[test_job]
    X_test = np.hstack([test.a, test.X])[idxs_test, :]
    y_test = test.targets.ravel()[idxs_test]

    # Test data for regressor prediction (without mapping)
    X_test_, y_test_ = translate_X_y(
        X_test, y_test, pca.centroids, n_knob_cols)

    # Test data for regressor prediction (with mapping)
    X_test__, y_test__ = translate_X_y(
        X_test, y_test, pca.altered_centroids, n_knob_cols)

    reg_ = reg.clone()
    err = reg_.MAPE(X_test_, y_test_)
    logging.info(
        "[Test job: {}] \t Error no-opt: {:.2f}".format(test_job, err))
    reg_.calibrate(X_calib_, y_calib_)
    err_cal = reg_.MAPE(X_test_, y_test_)
    logging.info("[Test job: {}] \t Error cal: {:.2f}".format(test_job,
                                                              err_cal))

    reg_ = reg.clone()
    assert np.abs(err - reg_.MAPE(X_test_, y_test_)) < 1e-10
    # Now let's use the data from mapping...
    err_mapping = reg_.MAPE(X_test__, y_test__)
    logging.info("[Test job: {}] \t Error map: {:.2f}".format(test_job,
                                                              err_mapping))
    reg_.calibrate(X_calib__, y_calib__)
    err_mapping_cal = reg_.MAPE(X_test__, y_test__)
    logging.info(
        "[Test job: {}] \t Error map_and_cal: {:.2f}".format(test_job,
                                                             err_mapping_cal))

    # err, err_cal, err_mapping, err_mapping_cal
    out_data['errs']['no-opt'][test_job].append(err)
    out_data['errs']['cal'][test_job].append(err_cal)
    out_data['errs']['map'][test_job].append(err_mapping)
    out_data['errs']['map_then_cal'][test_job].append(err_mapping_cal)


def train_model_and_evaluate(lods, out_data, seed=10):
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    n_knob_cols = len(lods.config['COLS_KNOBS'])

    nn_params = HYPER_PARAMS['nn_params']
    kpca_params = HYPER_PARAMS['kpca_params']

    tmp_trainval = lods.trainval
    tmp_shared_trainval = lods.shared_trainval

    if N_TRAIN_PER_JOB != -1:
        tmp_trainval = lods.trainval.get_x(N_TRAIN_PER_JOB)
    if N_SHARED_TRAIN_PER_JOB != -1:
        tmp_shared_trainval = lods.shared_trainval.get_x(N_SHARED_TRAIN_PER_JOB)

    if tmp_trainval is not None:
        logging.info("shape of remaining trainval (X): {}".format(
            tmp_trainval.X.shape))
    else:
        logging.info("tmp_trainval is None (perhaps because of get_x(0))")

    if tmp_shared_trainval is not None:
        logging.info(
            "shape of remaining shared trainval (X): {}".format(
                tmp_shared_trainval.X.shape))
    else:
        logging.info(
            "tmp_shared_trainval is None (perhaps because of get_x(0))")

    if tmp_trainval is None:
        # in case we're invoking dataset.get_x(0)
        ds_train = tmp_shared_trainval
    else:
        ds_train = tmp_trainval + tmp_shared_trainval

    X_train = np.hstack(
        [ds_train.a, ds_train.X, ds_train.Y])
    y_train = ds_train.targets.ravel()

    logging.info(
        "Fitting KPCA on data of shape: {}".format(
            X_train.shape))

    # Make PCA and fit on loaded data
    fit_t = time.time()
    pca = KernelPCA(**kpca_params)
    pca.altered_centroids = None
    logging.info("Fitting KPCA on data of shape: {}".format(ds_train.Y.shape))

    if ENCODING_STRATEGY == 'shared':
        shared_train = lods.shared_trainval.get_x(N_OBS)
        pca.fit(ds_train.Y)
        encods_shared = pca.transform(shared_train.Y)
        centroids = compute_centroids(encods_shared, shared_train.a)
    else:
        encods = pca.fit_transform(ds_train.Y)
        centroids = compute_centroids(encods, ds_train.a)
    pca.centroids = centroids  # thisis why I love Python! :-)
    fit_t = time.time() - fit_t
    logging.info("KPCA fitting time is: {} minutes and {} seconds".format(
        fit_t // 60, int(fit_t / 60)))

    # Adjust the X vector by transforming Y into job's centroid
    X, y = translate_X_y(X_train, y_train, pca.centroids, n_knob_cols)

    # Make and fit a NN Regressor
    logging.info(
        "Fitting regressor on data of shapes: {}, {}".format(
            X.shape, y.shape))
    reg = NNregressor(with_calibration=True, **
                      nn_params, random_state=seed, v1_compat_mode=True)
    reg.fit(X, y, log_time=True)
    training_mape = reg.MAPE(X, y)
    logging.info("Training Error: {:.2f}%".format(training_mape))
    out_data['training_errs'].append(training_mape)

    if ENCODING_STRATEGY == 'shared':
        observed_traces = lods.shared_traincomplement.get_x(N_OBS)
    else:
        observed_traces = lods.traincomplement.get_x(N_OBS)

    logging.info("observed_traces description: ")
    observed_traces.describe()

    observed_traces_slices = observed_traces.slice_by_job_id(
        alias_to_id=lods.alias_to_id)

    test_aliases = sorted(list(set(lods.test.a.ravel())))

    for test_job in observed_traces_slices:
        evaluate(test_job, pca, reg, lods, observed_traces,
                 observed_traces_slices, out_data, test_aliases)

    # out_data['kpcas'].append(pca)

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
