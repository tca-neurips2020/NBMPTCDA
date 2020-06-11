from sparkmodeling.common.lodatastruct import LODataStruct
import numpy as np
import os
from config import *


def test_eval_splits(lods, dataset):
    lods_mul_dataset = getattr(lods["mul"], dataset)

    for t in TEMPLATES:
        lods_t_dataset = getattr(lods[t], dataset)
        idxs_test = lods_mul_dataset.slice_by_aliases(
            by_job_id=True, alias_to_id=lods["mul"].alias_to_id)
        idxs_test_in_mul = [idxs_test[job] for job in TEST_WORKLOADS[t]]
        idxs_test_in_mul = [e for l in idxs_test_in_mul for e in l]

        mul_test_targets = lods_mul_dataset.targets[idxs_test_in_mul].ravel()
        t_test_targets = lods_t_dataset.targets.ravel()

        assert(
            np.sum(
                np.abs(t_test_targets - mul_test_targets)) <
            1e-10)

        print("Template {} - OK".format(t))
    print("*** test_eval_jobs_splits: PASS for dataset: ({})".format(dataset))


def test_eval_jobs(lods, dataset):
    lods_mul_dataset = getattr(lods['mul'], dataset)

    tc_jobs_mul = set([lods['mul'].alias_to_id[x]
                       for x in lods_mul_dataset.a.ravel()])
    for t in TEMPLATES:
        lods_t_dataset = getattr(lods[t], dataset)
        tc_jobs_t = set([lods[t].alias_to_id[x]
                         for x in lods_t_dataset.a.ravel()])
        assert tc_jobs_mul & set(TEMPLATES[t]) == tc_jobs_t
        assert tc_jobs_t & set(TEMPLATES[t]) == tc_jobs_t
        print("Template {} - OK".format(t))
    print("*** test_eval_jobs: PASS for dataset ({})".format(dataset))


def test_training_jobs(lods, dataset):
    lods_mul_dataset = getattr(lods['mul'], dataset)

    training_jobs_mul = set([lods['mul'].alias_to_id[x]
                             for x in lods_mul_dataset.a.ravel()])
    for t in TEMPLATES:
        lods_t_dataset = getattr(lods[t], dataset)
        training_jobs_t = set([lods[t].alias_to_id[x]
                               for x in lods_t_dataset.a.ravel()])

        assert training_jobs_mul & training_jobs_t == training_jobs_t

        s = set(training_jobs_t)  # copy
        non_test_workloads = list(set(TEMPLATES[t]) - set(TEST_WORKLOADS[t]))
        for e in non_test_workloads:
            s.add(e)

        assert s == training_jobs_mul
        print("Template {} - OK".format(t))
    print("*** test_training_jobs: PASS for dataset ({})".format(dataset))


def test_training_splits(lods, dataset):
    lods_mul_dataset = getattr(lods['mul'], dataset)
    for t in TEMPLATES:
        lods_t_dataset = getattr(lods[t], dataset)
        training_jobs_t = set([lods[t].alias_to_id[x]
                               for x in lods_t_dataset.a.ravel()])

        idxs_trainval = lods_mul_dataset.slice_by_aliases(
            by_job_id=True, alias_to_id=lods["mul"].alias_to_id)
        idxs_trainval_in_mul = [idxs_trainval[i] for i in training_jobs_t]
        idxs_trainval_in_mul = [e for l in idxs_trainval_in_mul for e in l]

        mul_trainval_targets = lods_mul_dataset.targets[idxs_trainval_in_mul].ravel(
        )
        t_trainval_targets = lods_t_dataset.targets.ravel()
        assert(
            np.sum(
                np.abs(
                    np.sort(t_trainval_targets) - np.sort(mul_trainval_targets)))
            < 1e-10)

        print("Template {} - OK".format(t))
    print("*** test_training_splits: PASS for dataset ({})".format(dataset))


def main():
    print("Loading LODS...")
    lods = {}
    lods_mul = LODataStruct.load_from_file(os.path.join(
        OUTPUT_FOLDER, "lods_mul.bin"), autobuild=True)
    lods["mul"] = lods_mul
    for t in TEMPLATES:
        lods[t] = LODataStruct.load_from_file(os.path.join(
            OUTPUT_FOLDER, "lods_{}.bin".format(t)), autobuild=True)
    print("LODS loaded, starting tests...")

    # 1) Make sure test data the same between LODS_X and LODS_mul
    test_eval_jobs(lods, 'test')
    test_eval_splits(lods, 'test')

    # 2) Make sure observation data is the same between LODS_X and LODS_mul
    test_eval_jobs(lods, 'traincomplement')
    test_eval_splits(lods, 'traincomplement')

    # 3) Make sure training data points in LODS_X also appear in LODS_mul
    test_training_jobs(lods, "trainval")
    test_training_splits(lods, "trainval")


if __name__ == "__main__":
    main()
