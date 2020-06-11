import numpy as np
import os
from sparkmodeling.autoencoder.aefancy import FancyAutoEncoder
import logging
from config import ENCODING_SIZE
from scipy.spatial.distance import cdist


def get_encodings(labels, centroids):
    """ Given a labels array and centroids dictionary, returns the encodings
    (centroids) corresponding to the current labels.
    """
    encodings = list(map(lambda x: centroids[int(x)], list(labels)))
    encodings = np.asarray(encodings)
    return encodings


def translate_X_y(X, y, centroids, n_knob_cols):
    """ Translates X, y to a format acceptable by a regressor
    """
    labels = X[:, 0]
    configurations = X[:, 1:1 + n_knob_cols]
    encodings = get_encodings(labels, centroids)
    X_ = np.hstack([configurations, encodings])
    y_ = np.ravel(y)
    return X_, y_


def persist_data(data, output_fp):
    np.save(output_fp, data)


def extract_encoding(extractor, trace, use_full_trace=True):
    observed_a, observed_X, observed_Y = trace
    if use_full_trace:
        xx = np.hstack([observed_a, observed_X, observed_Y])
    else:
        xx = observed_Y
    if np.ndim(xx) == 1:
        raise NotImplementedError
        # missing_centroid = extractor.transform(xx[np.newaxis, :])
    else:
        missing_centroid = extractor.transform(xx)
        logging.debug("before shape of centroid: {}".format(
            np.shape(missing_centroid)))
        missing_centroid = np.mean(missing_centroid, axis=0).ravel()
        logging.debug(
            "after shape of centroid (after mean): {}".format(
                np.shape(missing_centroid)))
        assert np.shape(missing_centroid)[0] == ENCODING_SIZE
    logging.debug("shape of input to autoencoder: {}".format(xx.shape))
    logging.debug(
        "shape of extracted centroid: {}".format(
            missing_centroid.shape))
    logging.debug("extracted centroid: {}".format(missing_centroid))
    alias = int(observed_a[0])
    return alias, missing_centroid


def extract_and_set_encoding(autoencoder, trace):
    """
    Extracts encoding from trace and sets it as the centroid of the concerned
    job.
    """
    alias, missing_centroid = extract_encoding(autoencoder, trace)
    autoencoder.centroids[alias] = missing_centroid


def extract_encoding_and_map_to_nearest(
        pca, trace, alias_to_id, banned_aliases,
        within_template=False, metric='euclidean'):
    """
    Extracts encoding from the given trace using a trained autoencoder and map
    the job to its nearest neighbor

    Returns the id of the proxy job (to which mapping was done)
    """
    _, _, debug = trace
    centroids_copy = pca.centroids.copy()
    print("Trace's Y shape: {}".format(debug.shape))

    # Step1: Extract encoding from the input trace
    alias, missing_centroid = extract_encoding(pca, trace, use_full_trace=False)
    # we don't set it here, because we want to borrow before we set it...
    if alias in centroids_copy:
        del centroids_copy[alias]
    print("missing centroid shape: {}".format(np.shape(missing_centroid)))

    # Step2: Calculate distances to other workloads
    if within_template:
        # Filter out the jobs from which we would like to do the mapping...
        raise NotImplementedError
    else:
        # We can search through all jobs
        aliases_to_check = sorted(list(centroids_copy.keys()))

        # Make sure this workload has not been seen before
        assert alias not in aliases_to_check

        # Filter out from aliases to check all test aliases (banned) in order
        # not to map to a previously evaluated test workload...
        aliases_to_check = [a for a in aliases_to_check
                            if a not in banned_aliases]

    centroids_to_compare = []
    for a in aliases_to_check:
        centroids_to_compare.append(centroids_copy[a])
    centroids_to_compare = np.vstack(centroids_to_compare)

    if metric == 'euclidean':
        distances = cdist(centroids_to_compare,
                          missing_centroid[np.newaxis]).squeeze()
        proxy_a = aliases_to_check[np.argmin(distances)]
    elif metric == 'cosine':
        similarities = cdist(centroids_to_compare, missing_centroid[np.newaxis],
                             metric='cosine').squeeze()
        proxy_a = aliases_to_check[np.argmax(similarities)]
    print(
        "{} ---> proxy job: {}".format(alias_to_id[alias], alias_to_id[proxy_a]))

    # Now let's borrow the encoding from this job
    if pca.altered_centroids is None:
        pca.altered_centroids = {}
    pca.altered_centroids[alias] = centroids_copy[proxy_a].copy()

    return alias_to_id[proxy_a]


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


def compute_centroids(encodings, labels):
    counts = {}
    centroids = {}
    encodings = encodings.copy()
    for i, encoding in enumerate(encodings):
        key = int(labels[i])
        if key in centroids:
            centroids[key] += encoding
            counts[key] += 1
        else:
            centroids[key] = encoding
            counts[key] = 1
    for key in centroids:
        centroids[key] /= counts[key]
    return centroids


def print_err_aggregates(errs):
    err_types = errs.keys()

    avg_errs = {}
    std_errs = {}
    for err_type in err_types:
        avg_errs[err_type] = {}
        std_errs[err_type] = {}
        for test_job in errs[err_types[0]].keys():
            avg_errs[err_type][test_job] = np.mean(
                errs[err_type][test_job])
            std_errs[err_type][test_job] = np.std(
                errs[err_type][test_job])

    for err_type in err_types:
        _all = []
        for test_job in avg_errs[err_type]:
            _all.append(avg_errs[err_type][test_job])

        logging.info("ERR TYPE: {}".format(err_type))
        logging.info(
            "[All Jobs] \t Mean error: {:.2f}% \t std dev: {:.2f}%".format(
                np.mean(_all),
                np.std(_all)))
