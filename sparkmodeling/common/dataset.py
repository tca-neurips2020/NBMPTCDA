import numpy as np
from ..common.utils import re_shape
from .utils import onehot


class Dataset:
    """
    A data structure object with attributes: a, X, Y and targets
    """

    def __init__(self, name=None):
        self.a = []
        self.X = []
        self.Y = []
        self.targets = []
        self.name = name

    def append(self, a, X, Y, targets):
        """Appends samples with aliases, X and Y to the current dataset.

        Note: after finalize is called, we're not allowed to call append
        anymore.

        Parameters:
        -----------
        a: array-like, shape = [n_samples, 1]
            Aliases of the jobs (or job ids) corresponding to each sample.
        X: array-like, shape = [n_samples, 4]
            Training vectors, where n_samples is the number of samples that
            has been added, and 4 corresponds to the 4 Spark streaming
            configuration parameters (Parallelism, BatchInterval,
            BlockInterval, InputRate).
        Y: array-like, shape = [n_samples, n_features] with n_features is the
        number of features corresponding to the metrics collected inside the
        traces. The matrix contains at this stage one column relative to our
        target: Delay(ms). It also contains metrics related to CPU, memory,
        IO and network.

        """
        self.a.append(a)
        self.X.append(X)
        self.Y.append(Y)
        self.targets.append(targets)

    def finalize(self):
        """Concatenates all appended samples and finalize the dataset.

        """
        if np.shape(self.a)[0] > 0:
            self.a = re_shape(np.concatenate(self.a))
            self.X = np.vstack(self.X)
            self.Y = np.vstack(self.Y)
            self.targets = np.vstack(self.targets)

    def extend_onehot(self, size):
        """Creates a new attribute `Xb` that combines X with onehot encodings.

        `Xb` stands for Xbaseline so that the features are the same features
        contained in X (Parallelism, BatchInterval, BlockInterval, InputRate)
        as well as new features representing the onehot encoding of the job's
        alias `a`.

        Parameter:
        ----------
        size: int
            The size of the onehot encoding vector. It must be greater than or
            equal to the total number of jobs we currently have.
        """
        aliases = list(map(lambda x: onehot(int(x), size=size), self.a))
        aliases = np.asarray(aliases)
        self.Xb = np.hstack([aliases, self.X])

    def fast_slice_by_job_id(self, alias_to_id):
        # New code: O(n)
        indexes = {}
        a_arr = self.a.ravel().astype(int)
        for i in range(len(a_arr)):
            if alias_to_id[a_arr[i]] in indexes:
                indexes[alias_to_id[a_arr[i]]].append(i)
            else:
                indexes[alias_to_id[a_arr[i]]] = [i]
        return indexes

    def fast_slice_by_aliases(self):
        # New code: O(n)
        indexes = {}
        a_arr = self.a.ravel().astype(int)
        for i in range(len(a_arr)):
            if a_arr[i] in indexes:
                indexes[a_arr[i]].append(i)
            else:
                indexes[a_arr[i]] = [i]
        return indexes

    def slice_by_aliases(self, by_job_id=False, alias_to_id=None, fast=True):
        """Slice the rows by the alias or job id
        """
        if fast:
            return self.fast_slice_by_aliases()

        # Old code: O(n^2)
        indexes = {}
        keys = np.unique(self.a).astype(int)
        for key in keys:
            idx = [i for i, a in enumerate(self.a) if np.asscalar(a) == key]
            if not by_job_id:
                indexes[key] = idx
            else:
                indexes[alias_to_id[key]] = idx
        return indexes

    def slice_by_job_id(self, alias_to_id, fast=True):
        """Slice the rows by the job id
        """
        if fast:
            return self.fast_slice_by_job_id(alias_to_id)
        # Old code: O(n^2):
        indexes = {}
        keys = np.unique(self.a).astype(int)
        for key in keys:
            idx = [i for i, a in enumerate(self.a) if np.asscalar(a) == key]
            indexes[alias_to_id[key]] = idx
        return indexes

    def get_instance_from_idxs(self, idxs):
        instance = Dataset()
        attributes = ['a', 'X', 'Y', 'targets']
        for att in attributes:
            setattr(instance, att, getattr(self, att)[idxs, :])
        return instance

    def get_x(self, x):
        """ Returns a new dataset object containing the first x data points from
        each job's data.
        This is going to be mainly used in the experiment in which we vary the
        number of training points.

        """
        slices = self.slice_by_aliases()
        idxs = []
        for a in slices:
            idxs.extend(slices[a][:x])
        if len(idxs) > 0:
            return self.get_instance_from_idxs(idxs)

        return None

    def get_min_n_configs(self):
        """
        Returns the minimum # of configs per job inside this dataset
        """
        slices = self.slice_by_aliases()
        _min = np.inf
        for a in slices:
            if len(slices[a]) < _min:
                _min = len(slices[a])
        return _min

    def get_max_n_configs(self):
        """
        Returns the maximum # of configs per job inside this dataset
        """
        slices = self.slice_by_aliases()
        _max = 0
        for a in slices:
            if len(slices[a]) > _max:
                _max = len(slices[a])
        return _max

    def stratified_split(self, first_split_size=.8):
        """
        Split this dataset into two datasets in a stratified way (as much as possible)
        according to different job ids...

        Returns two datasets split accordingly

        first_split_size: float
            Size of the first split
        """
        min_n_configs = self.get_min_n_configs()

        slices = self.slice_by_aliases()
        idxs_1 = []
        idxs_2 = []
        for a in slices:
            size = len(slices[a])
            s1 = int(first_split_size * size)
            s2 = size - s1
            if s2 <= 0:
                raise Warning(
                    "Warning: stratified_split has incurred a negative or zero size")
            idxs_1.extend(slices[a][:s1])
            idxs_2.extend(slices[a][s1:])

        assert len(idxs_1) > 0 and len(idxs_2) > 0
        dataset_1 = self.get_instance_from_idxs(idxs_1)
        dataset_2 = self.get_instance_from_idxs(idxs_2)

        return dataset_1, dataset_2

    def split(self, n1=None):
        if n1 is None:
            n1 = self.get_min_n_configs()//2

        slices = self.slice_by_aliases()
        idxs_1 = []
        idxs_2 = []
        for a in slices:
            idxs_1.extend(slices[a][:n1])
            idxs_2.extend(slices[a][n1:])

        assert len(idxs_1) > 0 and len(idxs_2) > 0
        dataset_1 = self.get_instance_from_idxs(idxs_1)
        dataset_2 = self.get_instance_from_idxs(idxs_2)

    def split_by_aliases(self, as_1, as_2):
        slices = self.slice_by_aliases()
        idxs_1 = []
        idxs_2 = []
        for a in slices:
            if a in as_1:
                idxs_1.extend(slices[a])
            elif a in as_2:
                idxs_2.extend(slices[a])
        assert len(idxs_1) > 0 and len(idxs_2) > 0

        dataset_1 = self.get_instance_from_idxs(idxs_1)
        dataset_2 = self.get_instance_from_idxs(idxs_2)

        return dataset_1, dataset_2

    def split_by_ids(self, ids_1, ids_2, id_to_alias):
        as_1 = [id_to_alias[i] for i in ids_1]
        as_2 = [id_to_alias[i] for i in ids_2]

        return self.split_by_aliases(as_1, as_2)

    def __add__(self, other):
        ds = Dataset()
        attributes = ['a', 'X', 'Y', 'targets']

        if other is None:
            for attribute in attributes:
                cp = getattr(self, attribute).copy()
                setattr(ds, attribute, cp)
            return ds

        else:
            try:
                for attribute in attributes:
                    merge = np.vstack(
                        [getattr(self, attribute),
                         getattr(other, attribute)]).copy()
                    setattr(ds, attribute, merge)
                return ds
            except Exception:
                print(
                    "[Error: can't add 2 datasets before finalize is invoked on them]")

    def describe(self):
        if self.name is None:
            name = ""
        else:
            name = self.name

        print("{:25}: \t min #configs/job: {} \t max #configs/job: {} \
            \t n_jobs: {} \t total points: {}".format(
            name,
            self.get_min_n_configs(),
            self.get_max_n_configs(),
            len(self.slice_by_aliases()),
            len(self.X)))

    def set_name(self, name):
        self.name = name
