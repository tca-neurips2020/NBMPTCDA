import os
import numpy as np
from time import time
from trainingutils import translate_X_y
from sparkmodeling.common.lodatastruct import LODataStruct
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


def get_pos(lods, idxs):
    return lods.trainval.get_instance_from_idxs(idxs)


def get_anchor(lods, idxs):  # anchor index in shared_trainval
    return lods.shared_trainval.get_instance_from_idxs(idxs)


def get_neg(lods, idxs):  # negative index in shared_trainval
    return lods.shared_trainval.get_instance_from_idxs(idxs)


def fetch_triplets(triplets_idxs, lods):
    # triplets_idxs: list of tuples.
    #      each tuple has 3 indexes: for anchor, pos and neg
    trips = np.array(triplets_idxs)
    anchor_idxs = trips[:, 0]
    pos_idxs = trips[:, 1]
    neg_idxs = trips[:, 2]

    anchor_ds = get_anchor(lods, anchor_idxs)
    pos_ds = get_pos(lods, pos_idxs)
    neg_ds = get_neg(lods, neg_idxs)

    return anchor_ds.Y, pos_ds.Y, neg_ds.Y, anchor_ds.X, pos_ds.X, neg_ds.X


def get_triplet_idxs(lods, with_shuffle=True, random_state=42):
    slices_tv = lods.trainval.slice_by_aliases()
    slices_stv = lods.shared_trainval.slice_by_aliases()
    anchors = lods.shared_trainval
    counter = 0
    triplets_idxs = []
    for alias_a in slices_stv:
        idxs_pos = slices_tv[alias_a]
        aliases_n = [key for key in slices_stv.keys() if key != alias_a]
        for idx_slice in range(
                len(slices_stv[alias_a])):  # looping over different configurations
            # now I have an anchor point
            idx_a = slices_stv[alias_a][idx_slice]
            config_a = anchors.X[idx_a, :]
            # negatives?
            idxs_neg = list(map(lambda alias, i: slices_stv[alias][i], aliases_n, [
                            idx_slice for i in range(len(aliases_n))]))
            for idx_pos in idxs_pos:
                for idx_neg in idxs_neg:
                    counter += 1
                    triplets_idxs.append((idx_a, idx_pos, idx_neg))
    if DEBUG_MODE:
        triplets_idxs = triplets_idxs[:100]
    if with_shuffle:
        return shuffle(triplets_idxs, random_state=random_state)
    return triplets_idxs


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
    def __init__(self, lods, triplet_idxs, q, out_data):
        Process.__init__(self)
        self.lods = lods
        self.q = q
        self.out_data = out_data
        self.triplet_idxs = triplet_idxs

    def run(self):
        import tensorflow as tf
        from sparkmodeling.autoencoder.tripletplusplus import TripletPlusPlus
        from sparkmodeling.nn.nnregressor import NNregressor
        lods = self.lods
        while True:
            seed = self.q.get()
            if seed == -1:
                break
            train_model_and_evaluate(
                lods, self.triplet_idxs, self.out_data, seed=seed, tf=tf,
                TripletPlusPlus=TripletPlusPlus, NNregressor=NNregressor)


def evaluate(test_job, autoencoder,
             reg, lods, observed_traces,
             observed_traces_slices, out_data, test_aliases=[]):
    # Get the trace(s) for this specific test job
    idx = observed_traces_slices[test_job]
    trace = observed_traces.a[idx], observed_traces.X[idx], \
        observed_traces.Y[idx]

    # Extract encoding from given trace(s) and trained autoencoder
    extract_and_set_encoding(autoencoder, trace)
    proxy = extract_encoding_and_map_to_nearest(
        autoencoder, trace, lods.alias_to_id, test_aliases)
    out_data['mappings'][test_job].append(proxy)

    # Test data (without encodings)
    test = lods.test
    slices_test = test.slice_by_job_id(lods.alias_to_id)
    idxs_test = slices_test[test_job]
    X_test = np.hstack([test.a, test.X])[idxs_test, :]
    y_test = test.targets.ravel()[idxs_test]

    # Test data for regressor prediction (without mapping)
    X_test_, y_test_ = translate_X_y(
        X_test, y_test, autoencoder.centroids)

    # Test data for regressor prediction (with mapping)
    X_test__, y_test__ = translate_X_y(
        X_test, y_test, autoencoder.altered_centroids)

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


def compute_centroids(autoencoder, lods, scheme='shared'):
    """computes centroids for training data"""
    centroids = {}
    if scheme == 'shared':
        ds = lods.shared_trainval.get_x(N_OBS)
    else:
        ds = lods.shared_trainval + lods.trainval
    slices = ds.slice_by_aliases()
    for a in slices:
        idxs = slices[a]
        encodings = autoencoder.transform(ds.Y[idxs, :])
        centroids[a] = np.mean(encodings, axis=0)

    autoencoder.centroids = centroids


def train_model_and_evaluate(
        lods, triplet_idxs, out_data, seed=10, tf=None, TripletPlusPlus=None,
        NNregressor=None):
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    n_knob_cols = lods.trainval.X.shape[1]
    # 2. train the autoencoder on the triplets

    autoencoder_params = HYPER_PARAMS['encoder_params']
    layer_sizes = [autoencoder_params['_nh']
                   ]*autoencoder_params['_nhlayers'] + [ENCODING_SIZE+n_knob_cols]
    del autoencoder_params['_nh']
    del autoencoder_params['_nhlayers']

    # Setting activations to be relu
    autoencoder_params['layer_sizes'] = layer_sizes
    autoencoder_params['activations'] = [
        'relu']*(len(autoencoder_params['layer_sizes']) - 1) + [None]

    # 561 metrics for streaming
    autoencoder_params['input_dim'] = lods.trainval.Y.shape[1]

    # n_knobs
    autoencoder_params['config_vec_size'] = n_knob_cols

    autoencoder = TripletPlusPlus(v1_compat_mode=True, **autoencoder_params)
    autoencoder.compile()
    autoencoder.centroids = None
    autoencoder.altered_centroids = None
    autoencoder.fit_idxs(triplet_idxs, fetch_triplets, lods, log_time=True)

    # 3. extract encodings for training workloads
    compute_centroids(autoencoder, lods, scheme=ENCODING_SCHEME)

    # 4. fetch training data for the regressor
    ds_train = lods.trainval + lods.shared_trainval
    X_ = np.hstack(
        [ds_train.a, ds_train.X])
    y_ = ds_train.targets.ravel()
    X, y = translate_X_y(X_, y_, autoencoder.centroids)

    # 5. Train a regressor on the training data
    nn_params = HYPER_PARAMS['nn_params']
    logging.info(
        "Fitting regressor on data of shapes: {}, {}".format(
            X.shape, y.shape))
    reg = NNregressor(**nn_params, random_state=seed)
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
        evaluate(test_job, autoencoder, reg, lods, observed_traces,
                 observed_traces_slices, out_data, test_aliases)

    # Append trained autoencoder information (with centroids) to output_data
    out_data['autoencoders'].append(autoencoder.get_persist_info())

    # Append trained regressor information to output_data
    out_data['regressors'].append(reg.get_persist_info())

    persist_data(copyDict(out_data), DATA_FNAME)


def get_lods(describe=False):
    # autobuild is set to false because persisted as built object.
    lods = LODataStruct.load_from_file(os.path.join(
        LODS_FOLDER_PATH, LODS_FNAME), autobuild=False)

    # Overwrite folder containing csvs and autobuild
    csv_folder = "../../../datasets/tpcx-bb/"
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

    # 1. fetch the triplets and fetch data for extracting
    #    encodings
    triplet_idxs = get_triplet_idxs(lods, with_shuffle=True, random_state=SEED)
    n_triplets = len(triplet_idxs)
    logging.info("Number of triplets: {}".format(n_triplets))
    triplet_idxs = triplet_idxs[:n_triplets//100]
    logging.info("Number of triplets (after trimming): {}".format(
        len(triplet_idxs)))

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
        p = OneWorkerProcess(lods, triplet_idxs, q, out_data)
        p.start()
        os.system("taskset -p -c %d %d" %
                  (((N_WORKERS*1+i) % os.cpu_count()), p.pid))
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
