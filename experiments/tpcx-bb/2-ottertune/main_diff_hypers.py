import os
import warnings
import getopt
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from cluster import KMeansClusters, create_kselection_model
from factor_analysis import FactorAnalysis
from gp import GPRNP
from gp_tf import GPRGD
from preprocessing import (Bin, get_shuffle_indices,
                           DummyEncoder, consolidate_columnlabels)
from config import *
from sparkmodeling.common.lodatastruct import LODataStruct
import logging
from copy import deepcopy
import socket
from multiprocessing import Process, Queue, Manager
import time

# The hyper-parameters that are provided in the config file are IGNORED
# inside this script:
# DEFAULT_LENGTH_SCALE
# DEFAULT_MAGNITUDE
# DEFAULT_RIDGE

rg_lengthscale = [1e-3, 1e-2, 1e-1, 1, 10]
rg_magnitude = [1e-3, 1e-2, 1e-1, 1, 10]
rg_ridge = [1e-3, 1e-2, 1e-1, 1, 10]

N_RUNS = 10  # overrides the one in CONFIG


def get_metrics_details(lods):
    """ Returns names of metric columns in the Y vector, the name of the
    ojective column and its index in this vector.
    """
    metrics_names = sorted(list(set(
        lods.cols_to_keep) - set(lods.config['COLS_KNOBS'] +
                                 lods.config['COLS_TODROP'])))

    # we only have latency as the target metric right now.
    objective_metric_name = lods.config['COLS_TARGETS'][0]
    idx_objective = np.where(np.asarray(metrics_names)
                             == objective_metric_name)[0][0]

    return {'labels': metrics_names,
            'objective_label': objective_metric_name,
            'objective_idx': idx_objective}


def read_lods(describe=False):
    # Autobuild is set to false because persisted as built object.
    lods = LODataStruct.load_from_file(os.path.join(
        LODS_FOLDER_PATH, LODS_FNAME), autobuild=False)

    # Overwrite folder containing csvs and autobuild
    csv_folder = "../../../datasets/tpcx-bb/"
    lods.folder = csv_folder
    lods._autobuild()

    # Notice that I didn't call minmaxscale, because Ottertune code does that.
    if describe:
        lods.describe()

    return lods


def fetch_workloads():
    """
    Read the workloads from the LODS definitions and unravel them into the
    format supported by Ottertune.

    """
    logging.info("Reading LODS...")
    lods = read_lods()
    logging.info("LODS read.")

    train_ds = lods.trainval + lods.shared_trainval
    test_ds = lods.test

    # Observe x data points from each test job
    if OBSERVED_SCHEME == "shared":
        observed_ds = lods.shared_traincomplement.get_x(N_OBS)
    elif OBSERVED_SCHEME == "not-shared":
        observed_ds = lods.traincomplement.get_x(N_OBS)
    else:
        raise NotImplementedError(
            "OBSERVED SCHEME '{}' is not Implemented.".format(observed_ds))

    train_slices = train_ds.slice_by_job_id(lods.alias_to_id)
    test_slices = test_ds.slice_by_job_id(lods.alias_to_id)
    observed_slices = observed_ds.slice_by_job_id(lods.alias_to_id)

    training_data = {}
    for job_id in train_slices:
        training_data[job_id] = {}
        training_data[job_id]['X_matrix'] = train_ds.X[train_slices[job_id], :]
        training_data[job_id]['y_matrix'] = train_ds.Y[train_slices[job_id], :]

    test_data = {}
    for job_id in test_slices:
        test_data[job_id] = {}
        test_data[job_id]['X_matrix'] = test_ds.X[test_slices[job_id], :]
        test_data[job_id]['y_matrix'] = test_ds.Y[test_slices[job_id], :]

    observed_data = {}
    for job_id in test_slices:
        observed_data[job_id] = {}
        observed_data[job_id]['X_matrix'] = observed_ds.X[observed_slices[job_id], :]
        observed_data[job_id]['y_matrix'] = observed_ds.Y[observed_slices[job_id], :]

    data = {'training': training_data,
            'test': test_data,
            'observed': observed_data,
            'metrics_details': get_metrics_details(lods)}
    return data


def run_workload_characterization(metric_data):
    # Performs workload characterization on the metric_data and returns
    # a set of pruned metrics.
    #
    # Parameters:
    #   metric_data is a dictionary of the form:
    #     - 'data': 2D numpy matrix of metric data (results x metrics)
    #     - 'rowlabels': a list of identifiers for the rows in the matrix
    #     - 'columnlabels': a list of the metric names corresponding to
    #                       the columns in the data matrix

    matrix = metric_data['data']
    columnlabels = metric_data['columnlabels']

    # Remove any constant columns
    nonconst_matrix = []
    nonconst_columnlabels = []
    for col, cl in zip(matrix.T, columnlabels):
        if np.any(col != col[0]):
            nonconst_matrix.append(col.reshape(-1, 1))
            nonconst_columnlabels.append(cl)
    assert len(nonconst_matrix) > 0, "Need more data to train the model"
    nonconst_matrix = np.hstack(nonconst_matrix)
    n_rows, n_cols = nonconst_matrix.shape

    # Bin each column (metric) in the matrix by its decile
    binner = Bin(bin_start=1, axis=0)
    binned_matrix = binner.fit_transform(nonconst_matrix)

    # Shuffle the matrix rows
    shuffle_indices = get_shuffle_indices(n_rows)
    shuffled_matrix = binned_matrix[shuffle_indices, :]

    # Fit factor analysis model
    fa_model = FactorAnalysis()
    # For now we use 5 latent variables
    fa_model.fit(shuffled_matrix, nonconst_columnlabels,
                 n_components=N_COMPONENTS)

    # Components: metrics * factors
    components = fa_model.components_.T.copy()

    # Run Kmeans for # clusters k in range(1, num_nonduplicate_metrics - 1)
    # K should be much smaller than n_cols in detK, For now max_cluster <= 20
    kmeans_models = KMeansClusters()
    kmeans_models.fit(components, min_cluster=1,
                      max_cluster=min(n_cols - 1, 20),
                      sample_labels=nonconst_columnlabels,
                      estimator_params={'n_init': 50})

    # Compute optimal # clusters, k, using gap statistics
    gapk = create_kselection_model("gap-statistic")
    gapk.fit(components, kmeans_models.cluster_map_)

    # Get pruned metrics, cloest samples of each cluster center
    pruned_metrics = kmeans_models.cluster_map_[
        gapk.optimal_num_clusters_].get_closest_samples()

    # Return pruned metrics
    return pruned_metrics


def map_workloads(observed_data, pruned_metrics_idxs, models, scalers):
    # loop over models, and try to predict the observed data for each
    # of the test workloads, and then

    od = {}
    for job in observed_data.keys():
        od[job] = {}
        od[job]['X_matrix'] = observed_data[job]['X_matrix'].copy()
        od[job]['y_matrix'] = observed_data[job]['y_matrix'][
            :, pruned_metrics_idxs].copy()

    scores = {}
    proxy_jobs = {}
    for test_workload in od:
        X_target = od[test_workload]['X_matrix']
        y_target = od[test_workload]['y_matrix']

        X_target = scalers['X_scaler'].transform(X_target)
        y_target = scalers['y_scaler'].transform(y_target)
        y_target = scalers['y_binner'].transform(y_target)

        scores[test_workload] = {}
        for training_workload in models.keys():
            predictions = np.empty_like(y_target)
            # one model per metric for each workload
            for j in range(len(models[training_workload])):
                predictions[:, j] = models[training_workload][j].predict(
                    X_target).ypreds.ravel()
            predictions = scalers['y_binner'].transform(predictions)
            dists = np.sqrt(np.sum(np.square(
                np.subtract(predictions, y_target)), axis=1))
            scores[test_workload][training_workload] = np.mean(dists)

        # Find the best (minimum) score
        best_score = np.inf
        for training_workload, score in list(scores[test_workload].items()):
            if score < best_score:
                best_score = score
                proxy_jobs[test_workload] = training_workload

    return proxy_jobs, scores


def train_gp_models(
        training_data, pruned_metrics_idxs, length_scale, magnitude, rdg):
    """ Trains separate model per workload
    """
    td = {}
    for job in training_data.keys():
        td[job] = {}
        td[job]['X_matrix'] = training_data[job]['X_matrix'].copy()
        td[job]['y_matrix'] = training_data[job]['y_matrix'][
            :, pruned_metrics_idxs].copy()

    # Stack all X & y matrices for preprocessing
    Xs = np.vstack([entry['X_matrix']
                    for entry in list(td.values())])
    ys = np.vstack([entry['y_matrix']
                    for entry in list(td.values())])

    # Scale the X & y values, then compute the deciles for each column in y
    X_scaler = StandardScaler(copy=False)
    X_scaler.fit(Xs)
    y_scaler = StandardScaler(copy=False)
    y_scaler.fit_transform(ys)
    y_binner = Bin(bin_start=1, axis=0)
    y_binner.fit(ys)
    del Xs
    del ys

    models = {}

    for workload_id, workload_entry in list(td.items()):
        # FIXME: this can be parallelized
        models[workload_id] = []
        X_workload = workload_entry['X_matrix']
        X_scaled = X_scaler.transform(X_workload)
        y_workload = workload_entry['y_matrix']
        y_scaled = y_scaler.transform(y_workload)
        # [KZ]: looping over the columns of the metrics
        for j, y_col in enumerate(y_scaled.T):
            # Using this workload's data, train a Gaussian process model
            # and then predict the performance of each metric for each of
            # the knob configurations attempted so far by the target.
            y_col = y_col.reshape(-1, 1)
            model = GPRNP(length_scale=length_scale,
                          magnitude=magnitude,
                          max_train_size=MAX_TRAIN_SIZE,
                          batch_size=BATCH_SIZE)
            model.fit(X_scaled, y_col, ridge=rdg)
            models[workload_id].append(model)

        scalers = {'X_scaler': X_scaler,
                   'y_scaler': y_scaler,
                   'y_binner': y_binner}
    return models, scalers


def train_and_evaluate(job_id, proxy_id, training_data,
                       observed_data, test_data, obj_idx,
                       length_scale, magnitude, rdg):
    """ Trains a separate model on mixed traces from current workload and
    mapped workload

    job_id: id the job on which we'd like to evaluate the model right now.
    proxy_id: id of the training job to which the workload is mapped.
    training_data: training data for training jobs: will be used to fetch the
                   proxy's job data...
    observed_data: observed data for the evaluation jobs.
    test_data: test data for the evaluation jobs.
    obj_idx: objective index in the metrics vector.

    (proxy_job is the job to which the current test workload has been mapped)
    """
    # Load mapped workload data
    X_workload = training_data[proxy_id]['X_matrix'].copy()
    y_workload = training_data[proxy_id]['y_matrix'][
        :, obj_idx].copy()

    # Target workload data (observed)
    X_target = observed_data[job_id]['X_matrix'].copy()
    y_target = observed_data[job_id]['y_matrix'][:, obj_idx].copy()

    # Target workload data on which we'll evaluate the model error...
    X_target_eval = test_data[job_id]['X_matrix'].copy()
    y_target_eval = test_data[job_id]['y_matrix'][:, obj_idx].copy()

    if np.ndim(y_workload) == 1:
        y_workload = np.expand_dims(y_workload, axis=1)

    if np.ndim(y_target) == 1:
        y_target = np.expand_dims(y_target, axis=1)

    if np.ndim(y_target_eval) == 1:
        y_target_eval = np.expand_dims(y_target_eval, axis=1)

    # Delete any rows that appear in both the workload data and the target
    # data from the workload data
    dups_filter = np.ones(X_workload.shape[0], dtype=bool)
    target_row_tups = [tuple(row) for row in X_target]
    for i, row in enumerate(X_workload):
        if tuple(row) in target_row_tups:
            dups_filter[i] = False
    X_workload = X_workload[dups_filter, :]
    y_workload = y_workload[dups_filter, :]

    # Combine target (observed) & workload (mapped) Xs for preprocessing
    X_matrix = np.vstack([X_target, X_workload])

    # Scale to N(0, 1)
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X_matrix)

    X_target_eval_scaled = X_scaler.transform(X_target_eval)

    # (KZ) Fitting scaler on both observed data as well as data from proxy job
    y_workload_scaler = StandardScaler()
    y_matrix = np.vstack([y_target, y_workload])
    y_scaled = y_workload_scaler.fit_transform(y_matrix)
    y_target_eval_scaled = y_workload_scaler.transform(y_target_eval)

    ###################### GP tensorflow training (fails) #####################
    # X_min = np.empty(X_scaled.shape[1])
    # X_max = np.empty(X_scaled.shape[1])

    # # Determine min/max for knob values
    # for i in range(X_scaled.shape[1]):
    #     col_min = X_scaled[:, i].min()
    #     col_max = X_scaled[:, i].max()

    #     X_min[i] = col_min
    #     X_max[i] = col_max

    # model = GPRGD(length_scale=DEFAULT_LENGTH_SCALE,
    #               magnitude=DEFAULT_MAGNITUDE,
    #               max_train_size=MAX_TRAIN_SIZE,
    #               batch_size=BATCH_SIZE,
    #               num_threads=NUM_THREADS,
    #               learning_rate=DEFAULT_LEARNING_RATE,
    #               epsilon=DEFAULT_EPSILON,
    #               max_iter=MAX_ITER,
    #               sigma_multiplier=DEFAULT_SIGMA_MULTIPLIER,
    #               mu_multiplier=DEFAULT_MU_MULTIPLIER)
    # model.fit(X_scaled, y_scaled, X_min, X_max, DEFAULT_RIDGE)
    # y_target_eval_pred = y_workload_scaler.inverse_transform(
    #     model.predict(X_target_eval_scaled).ypreds)
    ###########################################################################

    ###################### sklearn's RF as a regressor ########################
    if REG == "RF":
        raise NotImplementedError
    #     from sklearn.ensemble import RandomForestRegressor
    #     model = RandomForestRegressor(n_estimators=500)
    #     model.fit(X_scaled, y_scaled)
    #     y_target_eval_pred = y_workload_scaler.inverse_transform(
    #         model.predict(X_target_eval_scaled))
    #     y_train_pred = y_workload_scaler.inverse_transform(
    #         model.predict(X_scaled))
    #     training_mape = MAPE(y_matrix, y_train_pred)
    ###########################################################################
    # Numpy implementation of GP:
    elif REG == "GPNP":
        model = GPRNP(length_scale=length_scale,
                      magnitude=magnitude,
                      max_train_size=MAX_TRAIN_SIZE,
                      batch_size=BATCH_SIZE)
        model.fit(X_scaled, y_scaled, ridge=rdg)

        y_target_eval_pred = y_workload_scaler.inverse_transform(model.predict(
            X_target_eval_scaled).ypreds)  # we're returning the mean of the distribution here...
        logging.debug(
            "job {}: y_target_eval_pred: {}".format(
                job_id, y_target_eval_pred))

        y_train_pred = y_workload_scaler.inverse_transform(
            model.predict(X_scaled).ypreds)
        training_mape = MAPE(y_matrix, y_train_pred)
    else:
        raise NotImplementedError("This regressor is not implemented...")

    if np.ndim(y_target_eval_pred) > 1:
        y_target_eval_pred = np.squeeze(y_target_eval_pred)
    y_target_eval = np.squeeze(y_target_eval)

    logging.info("test workload: {} \t proxy: {} \t MAPE: {:.2f}%".format(
        job_id, proxy_id, MAPE(y_target_eval, y_target_eval_pred)))
    return MAPE(y_target_eval, y_target_eval_pred), training_mape


def main(run_id, length_scale, magnitude, rdg):
    logging.info(
        "MAIN is called with these params: run_id:{} \t length_scale:{} \t magnitude:{} \t rdg: {}".
        format(run_id, length_scale, magnitude, rdg))
    np.random.seed(run_id)
    logging.basicConfig(level=LOGGING_LEVEL)

    # 1. Fetch workloads
    data = fetch_workloads()

    # 2. Check which metrics to retain: pruned_metrics

    # # stacked Y vectors for all training workloads:
    stacked_metrics = np.vstack(
        [data['training'][job]['y_matrix'] for job in
         data['training'].keys()])
    col_labels = data['metrics_details']['labels']
    metric_data = {
        'data': stacked_metrics,
        'columnlabels': np.arange(len(col_labels))
    }
    logging.info("Running workload characterization to prune metrics...")

    pruned_metrics_idxs = run_workload_characterization(metric_data)
    pruned_metrics = [col_labels[i] for i in pruned_metrics_idxs]

    obj_lbl = data['metrics_details']['objective_label']
    obj_idx = data['metrics_details']['objective_idx']
    if obj_lbl not in pruned_metrics:
        logging.info(
            "pruned_metrics didn't contain the objective metric. Adding it now")
        pruned_metrics = [obj_lbl] + pruned_metrics
        pruned_metrics_idxs = [obj_idx] + pruned_metrics_idxs
        obj_idx_in_pruned = 0
    else:
        for i in range(len(pruned_metrics)):
            if pruned_metrics[i] == obj_lbl:
                obj_idx_in_pruned = i

    logging.info("Number of the pruned metrics: {}".format(len(pruned_metrics)))
    logging.info("pruned_metrics: {}".format(pruned_metrics))

    # 3. Train several models: one model for each training workload
    models, scalers = train_gp_models(
        data['training'],
        pruned_metrics_idxs, length_scale, magnitude, rdg)

    # 4. Map test workloads to training workloads
    proxy_jobs, scores = map_workloads(
        data['observed'], pruned_metrics_idxs, models, scalers)
    logging.info("Proxy jobs: {}".format(proxy_jobs))

    # 5. Train a separate model on mixed traces from the each of the test
    # workloads observed data and its mapped workload data...
    errors = {}
    training_errors = {}
    for test_job in proxy_jobs.keys():
        proxy_job = proxy_jobs[test_job]
        errors[test_job], training_errors[test_job] = train_and_evaluate(
            test_job, proxy_job, data['training'], data['observed'],
            data['test'], obj_idx,
            length_scale, magnitude, rdg)

    errs = []
    for test_job in errors.keys():
        errs.append(errors[test_job])

    logging.info("Errors: {}".format(errors))
    logging.info("Avg Errors across all jobs: {:.2f}%".format(np.mean(errs)))

    # FIXME: check if I'm missing dumping something important...
    run_info = {
        'pruned_metrics': pruned_metrics,
        'pruned_metrics_idxs': pruned_metrics_idxs,
        'proxy_jobs': proxy_jobs,
        'scores': scores,
        'errors': errors,
        'training_errors': training_errors
    }
    return run_info


def MAPE(y_true, y_pred):
    return 100 * np.mean(np.abs(y_true - y_pred) / y_true)


class oneWorkerProcess(Process):
    def __init__(self, q, all_data):
        Process.__init__(self)
        self.q = q
        self.all_data = all_data

    def run(self):
        while True:
            t = self.q.get()
            if t is None:
                break
            length_scale, magnitude, rdg = t
            runs_data = {'runs': {},
                         'CONFIG': CONFIG,
                         'length_scale': length_scale,
                         'magnitude': magnitude,
                         'ridge': rdg}
            for i in range(N_RUNS):
                runs_data['runs'][i] = main(i, length_scale, magnitude, rdg)
            signature = "{}_{}_{}".format(
                str(length_scale),
                str(magnitude),
                str(rdg))
            self.all_data[signature] = runs_data


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


if __name__ == "__main__":
    print("Number of runs: {}".format(N_RUNS))
    logging.basicConfig(level=getattr(logging, LOGGING_LEVEL))
    t0 = time.time()
    mng = Manager()
    all_data = mng.dict()
    q = Queue()
    for length_scale in rg_lengthscale:
        for magnitude in rg_magnitude:
            for rdg in rg_ridge:
                q.put((length_scale, magnitude, rdg))

    n_parallel = 10
    for i in range(n_parallel):
        q.put(None)

    processes = []
    for i in range(n_parallel):
        p = oneWorkerProcess(q, all_data)
        p.start()
        os.system("taskset -p -c %d %d" % (((i+0) % os.cpu_count()), p.pid))
        processes.append(p)

    for p in processes:
        p.join()

    tend = time.time()
    dt = tend-t0
    serialize_data = copyDict(all_data)
    serialize_data['running_duration'] = dt

    np.save("different_hypers_{}_{}.npy".format(
        OBSERVED_SCHEME, N_OBS), serialize_data)
    logging.info("Script took {} mins and {} seconds".format(dt//60, dt % 60))
