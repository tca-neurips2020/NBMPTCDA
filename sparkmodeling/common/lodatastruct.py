import os
import pandas as pd
import numpy as np
import json
import pickle
import warnings
from sparkmodeling.common.dataset import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.exceptions import DataConversionWarning


class LODataStruct:
    def __init__(self, test_jobs, lo_config_path, random_state=42,
                 test_size=0.8):
        """

        test_jobs: jobs for which we're splitting the data between
                    traincomplement and test.
        random_state: random seed
        test_size:

        Description of this class:
        Splits the data between trainval, test and train complement:
        trainval: contains indices of rows to be used from training jobs
                  (without the test job)
        test: contains indices of rows to be used for evaluation

        """
        self.random_state = random_state
        self.test_jobs = test_jobs  # now has become a list...
        self.read_config(lo_config_path)
        self.test_size = test_size
        self.empty_datasets()
        self.split_definitions = None
        self.jobs = None
        self.folder = None
        self.templates = None
        self.config_dict = None
        self.additional_jobs = []
        self.id_to_fname = None

    def empty_datasets(self):
        self.trainval = Dataset()
        self.traincomplement = Dataset()
        self.test = Dataset()
        self.shared_trainval = Dataset()
        self.shared_traincomplement = Dataset()
        self.datasets = [self.trainval, self.traincomplement, self.test,
                         self.shared_trainval, self.shared_traincomplement]

        self.scalers = {}

    def _get_scaling_ref(self):
        if self._si:
            return self.trainval + self.shared_trainval
        else:
            return self.trainval
        pass

    def extend_onehot(self):
        for dataset in self.datasets:
            dataset.extend_onehot(self.n_jobs)

    def minmaxscale(self, attr='X'):
        warnings.filterwarnings(
            "ignore", category=DataConversionWarning)
        scaler = MinMaxScaler()
        scaling_ref = self._get_scaling_ref()
        refattr = getattr(scaling_ref, attr)
        scaler.fit(refattr)
        for dataset in self.datasets:
            if len(getattr(dataset, attr)) > 0:
                dataset_attr = getattr(dataset, attr)
                dataset_attr = scaler.transform(dataset_attr)
                setattr(dataset, attr, dataset_attr)
        self.scalers[attr] = scaler

    def get_non_const_cols(self, df):
        to_keep = set()
        for col, keep in (df != df.iloc[0]).any().iteritems():
            if keep or col in self.config["COLS_KNOBS"] or \
                    col in self.config["COLS_TARGETS"] or \
                    'driver.FeatureEngineering' in col:
                to_keep.add(col)
        return to_keep

    def infer_valid_columns(self, jobs, folder):
        """
        Reads the csv files containing traces from different jobs and then
        return the name of columns that do not contain any NaN value in
        any trace file.

        jobs: ids of the jobs to be included within this data structure
        folder: folder containing csv files of the traces of the different
        jobs.
        """

        cols_to_keep = None
        for job in jobs:
            if self.id_to_fname is None:
                filepath = os.path.join(folder, "%d.csv" % job)
            else:
                filepath = os.path.join(
                    folder, "{}.csv".format(self.id_to_fname[job]))
            df = pd.read_csv(filepath)
            df.dropna(axis=1, inplace=True)
            if cols_to_keep is None:
                cols_to_keep = set(df.columns.values)
            else:
                cols_to_keep = cols_to_keep.intersection(set(df.columns.values))
            cols_to_keep_ = self.get_non_const_cols(df)
            cols_to_keep = cols_to_keep.intersection(cols_to_keep_)

        self.cols_to_keep = sorted(list(cols_to_keep))

    def read_jobs_data(
            self, jobs=None, folder=None, additional_jobs=[],
            id_to_fname=None):
        """
        jobs: ids of the jobs to be included within this data structure
        folder: folder containing csv files of the traces of the different jobs
        """
        if jobs is None:
            jobs = self.jobs
        if folder is None:
            folder = self.folder
        if id_to_fname is None:
            id_to_fname = self.id_to_fname
        else:
            self.id_to_fname = id_to_fname

        if len(additional_jobs) == 0:
            additional_jobs = self.additional_jobs
        else:
            self.additional_jobs = additional_jobs

        if folder is None:
            raise Exception(
                "Must provide folder before invoking read_jobs_data")
        elif jobs is None:
            raise Exception("Must provide jobs before invoking read_jobs_data")

        else:
            self.jobs = jobs
            self.folder = folder

        self.dataframes = {}
        self.infer_valid_columns(jobs + additional_jobs, folder)
        cols_to_keep = sorted(list(self.cols_to_keep))

        for col in self.config['COLS_KNOBS']:
            assert col in cols_to_keep

        for col in self.config['COLS_TARGETS']:
            assert col in cols_to_keep

        for job_id in jobs + additional_jobs:
            if id_to_fname is None:
                filepath = os.path.join(folder, "%d.csv" % job_id)
            else:
                filepath = os.path.join(folder, "{}.csv".format(
                    id_to_fname[job_id]))
            df = pd.read_csv(filepath)
            self.apply_preprocessing_rules(df, job_id)

        if os.path.exists(os.path.join(folder, "shared_configs.csv")):
            self.shared_configs_plan = pd.read_csv(
                os.path.join(folder, "shared_configs.csv"))
            for col in self.config['COLS_PREP_TR']:
                self.shared_configs_plan[col] = list(
                    map(lambda val: val[:-1], self.shared_configs_plan[col]))
            for col in self.config['COLS_PREP_TOINT']:
                self.shared_configs_plan[col] = list(
                    map(lambda val: int(val), self.shared_configs_plan[col]))
        else:
            self.shared_configs_plan = None

        self.create_id_alias_mapping(jobs)

    def create_id_alias_mapping(self, jobs):
        self.id_to_alias = {}
        self.alias_to_id = {}
        for i in range(len(jobs)):
            id = jobs[i]
            self.id_to_alias[id] = i
            self.alias_to_id[i] = id

    def apply_preprocessing_rules(self, df, job_id):
        df = df[self.cols_to_keep]
        for col in self.config['COLS_TARGETS']:
            df = df[df[col] > self.config['PREP_MIN_TARGET']]
            df = df[df[col] <= self.config['PREP_MAX_TARGET']]

        # Trim units in some of the columns:
        for col in self.config['COLS_PREP_TR']:
            df[col] = list(map(lambda val: val[:-1], df[col]))
        # Cast to integer
        for col in self.config['COLS_PREP_TOINT']:
            df[col] = list(map(lambda val: int(val), df[col]))

        # Remove duplicate rows
        df = df.groupby(self.config['COLS_KNOBS']).mean().reset_index()

        self.dataframes[job_id] = df

    def compute_split_definitions(self, imported_sd=None):
        """
        imported_sd: imported split definitions (from another LODS previously
                     constructed)
        """
        if len(self.jobs) == 0:
            raise Exception("Jobs data haven't yet been read")

        self.idxs_trainval = {}
        self.idxs_shared_trainval = {}
        self.idxs_traincomplement = {}
        self.idxs_shared_traincomplement = {}
        self.idxs_test = {}

        for job in self.jobs:
            a = self.id_to_alias[job]
            if not self._si:
                df = self.dataframes[job]
            else:
                df = self.dfs_not_intersecting[job]
                df_shared_configs = self.dfs_intersecting[job]

            if "COLS_METRICS" not in self.config or \
                    self.config['COLS_METRICS'] is None:
                y_cols = sorted(list(set(
                    self.cols_to_keep) - set(self.config['COLS_KNOBS'] +
                                             self.config['COLS_TODROP'])))
            else:
                y_cols = sorted(self.config['COLS_METRICS'])

            X = df[self.config['COLS_KNOBS']].values

            if self._si:
                X_shared = df_shared_configs[self.config['COLS_KNOBS']].values
                Y_shared = df_shared_configs[y_cols].values
                targets_shared = df_shared_configs[self.config
                                                   ['COLS_TARGETS']].values

            idxs = list(range(len(X)))

            if job not in self.test_jobs:
                # that's a training job
                is_intensive = job <= self.config["INTENSIVE_WORKLOAD_UPPER_ID"]
                if is_intensive or imported_sd is None:
                    self.idxs_trainval[job] = list(range(len(X)))
                elif imported_sd is not None and job in imported_sd['trainval']:
                    self.idxs_trainval[job] = imported_sd['trainval'][job]

                if self._si:
                    if imported_sd is not None and job in imported_sd['shared_trainval']:
                        self.idxs_shared_trainval[job] = imported_sd['shared_trainval'][job]
                    else:
                        self.idxs_shared_trainval[job] = list(
                            range(len(X_shared)))

            else:
                # It's a test job
                if imported_sd is not None and job in imported_sd[
                        'traincomplement'] and job in imported_sd['test']:
                    self.idxs_traincomplement[job] = \
                        imported_sd['traincomplement'][job]
                    self.idxs_test[job] = imported_sd['test'][job]
                else:
                    idxs_train, idxs_test = train_test_split(
                        idxs, random_state=self.random_state,
                        test_size=self.test_size)

                    self.idxs_traincomplement[job] = idxs_train
                    self.idxs_test[job] = idxs_test

                if self._si:
                    if imported_sd is not None and job in imported_sd['shared_traincomplement']:
                        self.idxs_shared_traincomplement[job] = imported_sd['shared_traincomplement'][job]
                    else:
                        self.idxs_shared_traincomplement[job] = list(
                            range(len(X_shared)))

        sds = {'trainval': self.idxs_trainval,
               'traincomplement': self.idxs_traincomplement,
               'test': self.idxs_test}

        if self._si:
            sds['shared_trainval'] = self.idxs_shared_trainval
            sds['shared_traincomplement'] = self.idxs_shared_traincomplement

        self.split_definitions = sds

    def get_split_definitions(self):
        return self.split_definitions

    def describe(self, with_vars=False):
        if with_vars:
            _vars = [self.random_state, self.test_jobs, self.test_size,
                     self.split_definitions, self.jobs, self.folder,
                     self.templates, self.config_dict]
            _var_names = ['random_state', 'test_jobs', 'test_size',
                          'split_definitions', 'jobs', 'folder',
                          'templates', 'config_dict']
            for vn, v in zip(_var_names, _vars):
                print("{}: {}".format(vn, v))

        if self._si:
            if self.templates is not None:
                for temp in self.templates:
                    print("Template [{}]".format(temp))
                    for job in self.stats[temp]:
                        n_int = self.stats[temp][job][0]
                        n_nonint = self.stats[temp][job][1]
                        somme = n_int + n_nonint
                        print("\t Job [{}]: # intersecting traces={} \t #non intersecting traces: {} \t Total: {}".format(
                            job, n_int, n_nonint, somme))
            else:
                for job in self.stats:
                    n_int = self.stats[job][0]
                    n_nonint = self.stats[job][1]
                    somme = n_int + n_nonint
                    print("\t Job [{}]: # intersecting traces={} \t #non intersecting traces: {} \t Total: {}".format(
                        job, n_int, n_nonint, somme))
        datasets = ['trainval', 'shared_trainval', 'test',
                    'traincomplement', 'shared_traincomplement']
        for ds_name in datasets:
            ds = getattr(self, ds_name)
            ds.name = ds_name
            ds.describe()

    @staticmethod
    def load_from_file(filepath, autobuild=False):
        with open(filepath, "rb") as file:
            instance = pickle.load(file)
        if autobuild:
            instance.read_jobs_data()
            instance.build(final_shuffle=instance._final_shuffle,
                           separate_intersections=instance._si,
                           templates=instance.templates)
        return instance

    def _autobuild(self):
        self.read_jobs_data()
        self.build(final_shuffle=self._final_shuffle,
                   separate_intersections=self._si,
                   templates=self.templates)

    def destroy_data(self):
        self.empty_datasets()
        self.dataframes = None
        self.dfs_intersecting = None
        self.dfs_not_intersecting = None

    def serialize(self, filepath, destroy=True):
        if destroy:
            self.destroy_data()
        pos = filepath.rfind("/")
        if pos != -1:
            if not os.path.exists(filepath[:pos]):
                try:
                    os.makedirs(filepath[:pos])
                except Exception:
                    print("An exception occured while trying to serialize object.")
            with open(filepath, "wb") as file:
                pickle.dump(self, file)

    def filter_in_dfs(self, jobs):
        return [j for j in jobs if j in self.dataframes]

    def _get_configurations_intersections_notemplates(
            self):
        shared_configs_plan = self.shared_configs_plan

        knobs = self.config['COLS_KNOBS']
        knobs_minus_ir = sorted(list(set(knobs) - set(['inputRate'])))

        self.dfs_intersecting = {}
        self.dfs_not_intersecting = {}
        self.stats = {}  # stats used in describe...

        shared_configs = self.dataframes[self.jobs[0]][knobs_minus_ir]
        for i in range(1, len(self.jobs)):
            job = self.jobs[i]
            shared_configs = pd.merge(shared_configs,
                                      self.dataframes[job][knobs_minus_ir],
                                      how='inner', on=knobs_minus_ir)

        jbs = list(set(self.jobs) - set(self.additional_jobs))
        for job in jbs:
            intersecting_traces = pd.merge(
                self.dataframes[job],
                shared_configs, how='inner', on=knobs_minus_ir)
            if shared_configs_plan is not None:
                non_intersecting_traces = self.dataframes[job][
                    pd.merge(self.dataframes[job], shared_configs_plan,
                             how='outer', on=knobs_minus_ir, indicator=True)
                    ['_merge'] == 'left_only']
            else:
                non_intersecting_traces = self.dataframes[job][
                    pd.merge(self.dataframes[job], shared_configs,
                             how='outer', on=knobs_minus_ir, indicator=True)
                    ['_merge'] == 'left_only']

            if shared_configs_plan is not None:
                assert(
                    len(non_intersecting_traces) + len(shared_configs_plan) >=
                    len(self.dataframes[job]))
                assert(
                    len(non_intersecting_traces) + len(shared_configs) <=
                    len(self.dataframes[job]))
            else:
                assert(
                    len(non_intersecting_traces) + len(shared_configs) ==
                    len(self.dataframes[job]))

            self.dfs_intersecting[job] = intersecting_traces
            self.dfs_not_intersecting[job] = non_intersecting_traces

            self.dfs_intersecting[job].sort_values(
                by=knobs, inplace=True)

            self.stats[job] = (
                len(intersecting_traces),
                len(non_intersecting_traces))

    def _get_configurations_intersections(
            self, templates):
        knobs = self.config['COLS_KNOBS']

        self.dfs_intersecting = {}
        self.dfs_not_intersecting = {}
        self.templates = templates
        self.stats = {}  # stats used in describe...
        for temp in templates:
            self.stats[temp] = {}
            temp_jobs = templates[temp]
            shared_configs = self.dataframes[temp_jobs[0]][knobs]
            for i in range(1, len(temp_jobs)):
                job = temp_jobs[i]
                shared_configs = pd.merge(shared_configs,
                                          self.dataframes[job][knobs],
                                          how='inner', on=knobs)

            temp_jobs = list(set(temp_jobs) - set(self.additional_jobs))

            for job in temp_jobs:
                self.stats[temp][job] = {}
                intersecting_traces = pd.merge(
                    self.dataframes[job],
                    shared_configs, how='inner', on=knobs)
                non_intersecting_traces = self.dataframes[job][
                    pd.merge(self.dataframes[job], shared_configs,
                             how='outer', on=knobs, indicator=True)
                    ['_merge'] == 'left_only']
                assert(
                    len(non_intersecting_traces) + len(shared_configs) ==
                    len(self.dataframes[job]))
                self.dfs_intersecting[job] = intersecting_traces
                self.dfs_not_intersecting[job] = non_intersecting_traces

                self.dfs_intersecting[job].sort_values(by=knobs, inplace=True)
                self.stats[temp][job] = (
                    len(intersecting_traces),
                    len(non_intersecting_traces))

    def build(self, imported_sd=None,
              final_shuffle=True,
              separate_intersections=False,
              templates=None):
        """
        separate_intersections: boolean, default (False). 
            Whether or not we need to separate the intersecting configurations
            (among configurations from all jobs belonging to the same template)
            into a separate dataset before building or mix them with other
            configurations...
        templates: dict
            Keys are going to be the job ids, and values are going to be lists
            with job ids within each template.
        """
        self._final_shuffle = final_shuffle
        if templates is not None:
            self.templates = templates
        else:
            templates = self.templates  # useful when reading serialized object.
        try:
            assert self.dataframes is not None
        except:
            raise Exception("read_jobs_data must be called before build")

        if separate_intersections:
            if self.templates is not None:
                self._get_configurations_intersections(templates=templates)
            else:
                self._get_configurations_intersections_notemplates()
            self._si = True
        else:
            self._si = False

        if self.split_definitions is None:
            self.compute_split_definitions(
                imported_sd=imported_sd)

        sds = self.split_definitions

        idxs_trainval = sds['trainval']
        idxs_traincomplement = sds['traincomplement']
        idxs_test = sds['test']

        if separate_intersections:
            idxs_shared_trainval = sds['shared_trainval']
            idxs_shared_traincomplement = sds['shared_traincomplement']

        if len(self.jobs) == 0:
            raise Exception("Jobs data haven't yet been read")
        X_trainval = []
        Y_trainval = []
        targets_trainval = []
        a_trainval = []

        for job in self.jobs:
            a = self.id_to_alias[job]
            if not separate_intersections:
                df = self.dataframes[job]
            else:
                df = self.dfs_not_intersecting[job]
                df_shared_configs = self.dfs_intersecting[job]

            if "COLS_METRICS" not in self.config or\
                    self.config['COLS_METRICS'] is None:
                y_cols = sorted(list(
                    set(self.cols_to_keep) - set(self.config['COLS_KNOBS'] +
                                                 self.config['COLS_TODROP'])))
            else:
                y_cols = sorted(self.config['COLS_METRICS'])

            X = df[self.config['COLS_KNOBS']].values
            Y = df[y_cols].values
            targets = df[self.config['COLS_TARGETS']].values

            if separate_intersections:
                X_shared = df_shared_configs[self.config['COLS_KNOBS']].values
                Y_shared = df_shared_configs[y_cols].values
                targets_shared = df_shared_configs[self.config
                                                   ['COLS_TARGETS']].values

            if job in idxs_trainval:
                # that's a training job!
                self.trainval.append(a * np.ones(len(idxs_trainval[job])),
                                     X[idxs_trainval[job]],
                                     Y[idxs_trainval[job]],
                                     targets[idxs_trainval[job]])

                if separate_intersections:
                    assert job in idxs_shared_trainval
                    self.shared_trainval.append(
                        a * np.ones(len(idxs_shared_trainval[job])),
                        X_shared[idxs_shared_trainval[job]],
                        Y_shared[idxs_shared_trainval[job]],
                        targets_shared[idxs_shared_trainval[job]])

            else:
                if job in idxs_test:
                    a_ = a * np.ones(len(idxs_test[job]))
                    self.test.append(
                        a_, X[idxs_test[job]],
                        Y[idxs_test[job]],
                        targets[idxs_test[job]])

                if job in idxs_traincomplement:
                    self.traincomplement.append(
                        a * np.ones(len(idxs_traincomplement[job])),
                        X[idxs_traincomplement[job]],
                        Y[idxs_traincomplement[job]],
                        targets[idxs_traincomplement[job]])
                    if separate_intersections:
                        assert job in idxs_shared_traincomplement
                        self.shared_traincomplement.append(
                            a * np.ones(len(idxs_shared_traincomplement[job])),
                            X_shared[idxs_shared_traincomplement[job]],
                            Y_shared[idxs_shared_traincomplement[job]],
                            targets_shared[idxs_shared_traincomplement[job]])

        for dataset in self.datasets:
            dataset.finalize()

        if final_shuffle:
            self.trainval.a, self.trainval.X, self.trainval.Y,\
                self.trainval.targets = shuffle(
                    self.trainval.a, self.trainval.X, self.trainval.Y,
                    self.trainval.targets, random_state=self.random_state)

    def read_config(self, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        self.config = config

    def set_config_dict(self, config_dict):
        self.config_dict = config_dict
