from sparkmodeling.common.lodatastruct import LODataStruct
from sparkmodeling.nn.embeddings import KerasEmbeddingRegressor
import numpy as np
import time
import os
import logging
from config import *
from multiprocessing import Manager, Process, Queue
import tensorflow as tf
import numpy as np


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


def copyDict(mdict):
    import multiprocessing
    copy = {}
    for key in mdict.keys():
        if isinstance(mdict[key], multiprocessing.managers.DictProxy):
            copy[key] = copyDict(mdict[key])
        elif isinstance(mdict[key], multiprocessing.managers.ListProxy):
            copy[key] = [el for el in mdict[key]]
        else:
            copy[key] = mdict[key]
    return copy


def get_lods(describe=False):
    # autobuild is set to false because persisted as built object.
    lods = LODataStruct.load_from_file(os.path.join(
        LODS_FOLDER_PATH, LODS_FNAME), autobuild=False)

    # Overwrite folder containing csvs and autobuild
    lods.id_to_fname = None  # backward compatibility for streaming
    csv_folder = "../../../datasets/streaming/"
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


class OneWorkerProcess(Process):
    def __init__(self, lods, q, data_container):
        Process.__init__(self)
        self.lods = lods
        self.q = q
        self.data_container = data_container

    def run(self):
        while True:
            seed = self.q.get()
            np.random.seed(seed)
            tf.compat.v1.set_random_seed(seed)
            if seed is None:
                break
            train_and_evaluate(self.lods, seed, self.data_container)


def train_and_evaluate(lods, seed, data_container):
    ds_train = lods.trainval + lods.shared_trainval

    X_em = np.hstack([ds_train.a, ds_train.X])
    y_em = ds_train.targets.ravel()

    len_knob_cols = len(lods.config['COLS_KNOBS'])

    er = KerasEmbeddingRegressor(VOCAB_SIZE, len_knob_cols,
                                 random_state=seed, compat_mode=True, **PARAMS)

    history = er.fit(X_em, y_em, log_time=True)

    training_error = er.MAPE(X_em, y_em)
    data_container['training_errors'].append(training_error)
    data_container['fit_durations'].append(er._last_fit_duration)
    data_container['regressor_clones'].append(er.get_persist_info())

    if OBSERVATION_SCHEME == "all":
        observed_data = lods.traincomplement.get_x(N_OBS)
    elif OBSERVATION_SCHEME == "shared":
        observed_data = lods.shared_traincomplement.get_x(N_OBS)

    tinf_0 = time.time()
    slices_observed = observed_data.slice_by_job_id(lods.alias_to_id)
    slices_test = lods.test.slice_by_job_id(lods.alias_to_id)

    for test_job in slices_observed:
        idxs = slices_observed[test_job]
        idxs_test = slices_test[test_job]
        logging.debug("Running inference for test job: {}".format(test_job))

        reg = er.clone()

        # Freezing the weights of the neural network (except the embeddings that are left as
        # degrees of freedom...)
        for layer in reg.keras_model.layers[5:]:
            layer.trainable = False

        X_obs = np.hstack([observed_data.a[idxs, :], observed_data.X[idxs, :]])
        y_obs = observed_data.targets[idxs, :].ravel()

        X_test = np.hstack(
            [lods.test.a[idxs_test, :],
             lods.test.X[idxs_test, :]])
        y_test = lods.test.targets[idxs_test, :].ravel()

        # checking test errors before incrementally training:
        data_container['errors_before_inc_training'][test_job].append(
            reg.MAPE(X_test, y_test))

        reg.patience = PATIENCE_INC
        reg.n_epochs = N_INC
        reg.early_stopping = ES_INC
        his = reg.fit(X_obs, y_obs)

        data_container['inc_training_errors'][test_job].append(
            reg.MAPE(X_obs, y_obs))
        data_container['errors'][test_job].append(reg.MAPE(X_test, y_test))

    tinf_end = time.time()
    inference_duration = tinf_end - tinf_0

    errs = []
    for test_job in slices_observed:
        errs.append(np.mean(data_container['errors'][test_job]))
    logging.info(
        "Test errors: {:.2f} +- {:.2f}".format(np.mean(errs), np.std(errs)))

    logging.info("Inference duration: {} and {} seconds".format(
        inference_duration//60, int(inference_duration % 60)))

    data_container['inference_durations'].append(inference_duration)

    np.save(DATA_FNAME, copyDict(data_container))


def make_distributed_dicts(manager, test_jobs):
    errors = manager.dict()
    errors_before_inc_training = manager.dict()
    inc_training_errors = manager.dict()
    training_errors = manager.list()
    fit_durations = manager.list()
    inference_durations = manager.list()
    regressor_clones = manager.list()

    for test_job in test_jobs:
        errors[test_job] = manager.list()
        inc_training_errors[test_job] = manager.list()
        errors_before_inc_training[test_job] = manager.list()

    data = manager.dict()
    data['errors_before_inc_training'] = errors_before_inc_training
    data['training_errors'] = training_errors
    data['inc_training_errors'] = inc_training_errors
    data['errors'] = errors
    data['fit_durations'] = fit_durations
    data['inference_durations'] = inference_durations
    data['regressor_clones'] = regressor_clones

    return data


def main():
    t0 = time.time()
    logging.basicConfig(level=LOGGING_LEVEL)
    lods = get_lods(describe=True)

    lods_copy = get_lods()
    lods_copy.destroy_data()

    test_aliases = list(set(lods.test.a.ravel()))
    test_jobs = list(map(lambda a: lods.alias_to_id[a], test_aliases))

    manager = Manager()
    data_container = make_distributed_dicts(manager, test_jobs)

    data_container['lods'] = lods_copy
    data_container['CONFIG'] = get_config_dict()

    q = Queue()
    for i in range(N_RUNS):
        q.put(i)
    for i in range(N_WORKERS):
        q.put(None)

    processes = []
    for i in range(N_WORKERS):
        wp = OneWorkerProcess(lods, q, data_container)
        wp.start()
        processes.append(wp)

    for p in processes:
        p.join()

    tf = time.time()
    running_time = tf - t0
    aux = copyDict(data_container)
    aux['running_time'] = running_time
    np.save(DATA_FNAME, aux)


if __name__ == "__main__":
    main()
