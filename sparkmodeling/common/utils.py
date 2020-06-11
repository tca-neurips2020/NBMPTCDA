import json
import requests
import numpy as np
import tensorflow as tf
import os


class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self


def swap_cols(X, i, j):
    # Inplace swap columns i and j of matrix X
    aux = X[:, i].copy()
    X[:, i] = X[:, j]
    X[:, j] = aux


def mprint(l, precision=3):
    nl = []
    for e in l:
        ee = (str(e)).split('.')
        ne = ee[0] + '.' + ee[1][:precision]
        nl.append(float(ne))
    print(nl)


def identity_function(tensor):
    return tensor


def beep():
    os.system("cvlc --play-and-exit ~/notification.mp3 &")


def weight_variable(shape, trainable=True, init_std=0.1):
    initial = tf.truncated_normal(shape, stddev=init_std)
    return tf.Variable(initial, trainable=trainable)


def bias_variable(shape, trainable=True, init=0.1):
    initial = tf.constant(init, shape=shape)
    return tf.Variable(initial, trainable=trainable)


def weight_variable2(shape, trainable=True, r=1):
    initial = tf.random_uniform(shape, minval=-r, maxval=r)
    return tf.Variable(initial, trainable=trainable)


def bias_variable2(shape, trainable=True):
    initial = tf.constant(1e-10, shape=shape)
    return tf.Variable(initial, trainable=trainable)


def get_feed_dict(a, X, Y, placeholders):
    a_input, X_input, Y_true = placeholders
    feed_dict = {a_input: a, X_input: X, Y_true: Y}
    return feed_dict


def re_shape(a):
    return np.reshape(a, [np.shape(a)[0], 1])


def rescale_mean_std(X, mean, std):
    EPSILON = 1e-10
    return (X - mean) / (EPSILON + std)


def rescale_max_min(X, mean=None, mini=None, maxi=None):
    EPSILON = 1e-10
    if mean is None and mini is None and maxi is None:
        return (X - np.mean(X, axis=0)) / (EPSILON + np.max(X, axis=0) - np.min(X, axis=0))
    return (X - mean) / (EPSILON + maxi - mini)


def repeat(arr, ntimes):
    return np.repeat(arr, ntimes, axis=0)


def vprint(verbose=True, *args):
    if verbose:
        print(*args)


def onehot(job_id, job_to_alias=None, size=6):
    """Returns onehot vector representation of job using its alias.

    Parameters:
    ----------
    job: int
        The id of the job to encode
    job_to_alias: dict, optional (default=None)
        A map from job id to alias such that aliases for all jobs start at 0.
        If job_to_alias is None, then job id must be between 0 and size - 1
    size: int, optional (default=6)
        The total number of jobs that are going to be encoded
    """

    if job_to_alias is None:
        return np.eye(size)[job_id]
    return np.eye(size)[job_to_alias[job_id]]


def inv_onehot(vector, alias_to_job=None):
    """Returns the job id corresponding to the encoded vector

    Parameters:
    ----------
    vector: array-like, shape=[size]
        The onehot vector representation of a job
    alias_to_job: dict, optional (defaul=None)
        A map from alias to job_id such that aliases for all jobs start at 0
        If job_to_alias is None, then the returned job id will be between 0
        and size - 1
    """
    alias = int(np.asscalar(np.where(vector == 1)[0]))
    if alias_to_job is None:
        return alias
    return alias_to_job[alias]


def slice_onehot_data(X, y, indices, alias_to_job=None):
    """Creates a map for X and a map for y after slicing data by job id.

    This function is useful when we want to evaluate a previously trained
    regressor, on parts of the training or test matrix to see how well it can
    predict latency corresponding to a particular job. Generally, we use this
    function to calculate a detailed version of MAPE.
    Note: This function only works when jobs are onehot encoded

    Parameters:
    -----------
    X: array-like matrix, shape=[n_samples, n_features]
        The data matrix where columns indexed by `indices` contain the encoding
        of the jobs.
        - n_samples is the number of samples in the matrix X
        - n_features is the number of features and is equal to the dimension
        of the encoding plus 4 (the 4 features corresponding to Parallelism,
        BatchInterval, BlockInterval and InputRate)
    y: array-like, shape=[n_samples]
        The target array of size n_samples.
    indices: list
        The indices of the columns that represent the encodings
    alias_to_job: dict, optional (default=None)
        A map from alias to job_id such that aliases for all jobs start at 0
        If job_to_alias is None, then the returned job id will be between 0
        and count(jobs) - 1
    Returns:
    --------
    X_: dict
        A dictionary mapping job ids to array-like matrices each corresponding
        to the slice of the input matrix X related to the job id.

    y_: dict
        A dictionary mapping job ids to arrayseach corresponding to the slice
        of the input array y related to the job id.

    """
    X_, y_ = {}, {}
    for i in range(len(X)):
        job = inv_onehot(X[i, indices], alias_to_job=alias_to_job)
        if job not in X_:
            X_[job] = []
            y_[job] = []
        X_[job].append(X[i, :])
        y_[job].append(y[i])
    for job, _ in X_.items():
        X_[job] = np.vstack(X_[job])
        y_[job] = np.asarray(y_[job])
    return X_, y_


def row_in_matrix(row, matrix):
    """Checks whether a numpy array is one of the rows of a matrix

    Parameters:
    -----------
    row: array-like, shape: (n_cols,)
        The row to be searched for inside the matrix
    matrix: array-like, shape: (n_rows, n_cols)
        The matrix inside which we search for the row
    """
    for r in matrix:
        if np.linalg.norm(row - r) < 1e-20:
            return True
    return False


def locate_row_in_matrix(row, matrix):
    """Returns the index of the row of the matrix or -1

    Parameters:
    -----------
    row: array-like, shape: (n_cols,)
        The row to be searched for inside the matrix
    matrix: array-like, shape: (n_rows, n_cols)
        The matrix inside which we search for the row
    """
    for i, r in enumerate(matrix):
        if np.linalg.norm(row - r) < 1e-15:
            return i
    return -1


def locate_rows_in_matrix(row, matrix):
    """Returns a list containing the indexes of the row in the matrix

    Parameters:
    -----------
    row: array-like, shape: (n_cols,)
        The row to be searched for inside the matrix
    matrix: array-like, shape: (n_rows, n_cols)
        The matrix inside which we search for the row
    """
    idxs = []
    for i, r in enumerate(matrix):
        if np.linalg.norm(row - r) < 1e-15:
            idxs.append(i)
    return idxs


def train_test_split_(*arrays, test_size=.2, shuffle=True,
                      indexes=None, random_state=42):
    """Split arrays or matrices into random train and test subsets


    The difference with sklearn's implementation is that it's possible to
    preserve the indexes used in splitting for a future call of the same
    method but on new data of the same length/shape[0] as the current data

    Parameters:
    -----------
    *arrays: sequence of indexables with same length / shape[0]
        Allowed inputs are numpy arrays
    test_size: float
        It should be between 0.0 and 1.0 and represent the proportion of the
        dataset to include in the test split.
    shuffle: boolean, optional (default=True)
        Whether or not to shuffle data before splitting
    indexes: list or array-like
        List of integers representing the new ordering of the samples on which
        the splitting will be applied. Ignored if shuffle is False
    random_state: int
        Pseudo-random number generator state used for random sampling. Ignored
        if shuffle is set to False, or if indexes are not None

    Returns:
    --------
    to_return: list, length=2 * len(arrays) + 1
        List containing train-test splits of inputs as well as the indexes used
        right before splitting


    Example:
    --------
    - if shuffle is True and indexes is None: Then random state is going to be
     taken into consideration
    - if shuffle is True and indexes are given: Then random state is going to
     be ignored
    - if shuffle is False, regardless of the values of indexes and random_state
    data is going to be spliltted without any shuffling.

    Our usage:
    first call the function with shuffle=True and indexes=None, it will
    generate an index, and it's going to return the indexes with the
    first data that has been split.
    Then with some other data, call again this function with shuffle=True
    but this time, provide the indexes returned by the previous call as an
    argument for the new call.

    """
    n_rows = len(arrays[0])
    barrier = int(n_rows * (1 - test_size))

    if shuffle:
        if indexes is None:
            indexes = list(range(n_rows))
            np.random.seed(random_state)
            indexes = np.random.choice(indexes, n_rows, replace=False)
    else:
        indexes = list(range(n_rows))

    to_return = []
    for arr in arrays:
        if np.ndim(arr) == 1:
            arr = arr[indexes]
            arr_train = arr[:barrier]
            arr_test = arr[barrier:]
        else:
            arr = arr[indexes, :]
            arr_train = arr[:barrier, :]
            arr_test = arr[barrier:]
        to_return.append(arr_train)
        to_return.append(arr_test)
    to_return.append(indexes)
    return to_return


def identity_tensor(tensor):
    return tensor


def create_dir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)


def my_onehot(x, size=6):
    return np.eye(size)[int(x)]
