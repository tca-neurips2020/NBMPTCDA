import numpy as np
from sparkmodeling.common.lodatastruct import LODataStruct
from sparkmodeling.common.utils import row_in_matrix
from config import TEMPLATES, TEST_WORKLOADS, OUTPUT_FOLDER
from config import DATA_FOLDER, CONFIG_PATH
from helpers_make_lods import make_LODatastruct_mul
import os

# TODO: I should verify that such tests also won't fail if I re-read the serialized lods object


# Only keeping workloads within the template if they appear in the dataset.
def assert_config_match(lods, dataset):
    """
    Asserts that configurations are matching across different workloads of
    the same template
    (useful for checking shared_trainval and shared_traincomplement made
    with lodatastruct_mul)
    """
    slices = dataset.slice_by_job_id(lods.alias_to_id)
    # Remove jobs not appearing in the dataset (these must be test jobs)
    templates_ = {
        key: [job for job in TEMPLATES[key] if job in slices]
        for key in TEMPLATES}
    for temp in templates_:
        first_job = templates_[temp][0]  # first job in template
        l = len(slices[first_job])
        for i in range(1, len(templates_[temp])):
            job = templates_[temp][i]
            assert len(slices[job]) == l
            for k in range(l):
                assert np.sum(
                    np.abs(
                        dataset.X[slices[job][k],
                                  :] - dataset.X[slices[first_job][k],
                                                 :])) < 1e-10


def assert_no_shared_in_test(lods, shared_dataset="shared_trainval"):
    """ Asserts that none of the shared configurations appear in test.

    Since shared configurations are repeated across workloads from the same template,
    we can first take them from any of the workloads within the template.

    Then, we loop over test jobs and make sure that none of these shared configurations appear in test job.
    """
    slices_shared = getattr(
        lods, shared_dataset).slice_by_job_id(
        lods.alias_to_id)
    slices_test = lods.test.slice_by_job_id(lods.alias_to_id)

    for key in TEMPLATES:
        jobs_within_template = TEMPLATES[key]
        shared_configs = None

        # Get shared configurations within a particular template
        for job in jobs_within_template:
            if job in slices_shared:
                idxs = slices_shared[job]
                shared_configs = getattr(lods, shared_dataset).X[idxs]
                break

        # Make sure that none of these configurations appear in test workloads
        for job in jobs_within_template:
            if job in slices_test:
                configs = lods.test.X[slices_test[job], :]
                for i in range(len(shared_configs)):
                    assert not row_in_matrix(shared_configs[i, :], configs)
    print("Assertion OK for no shared configurations in test")


def check_shared_configs_consistency(lods):
    """Checking shared configs consistency when changing the number of points.
    """
    shared_trainvals = [lods.shared_trainval.get_x(
        x) for x in [1, 10, 32]] + [lods.shared_trainval]
    shared_traincomplements = [lods.shared_traincomplement.get_x(x) for x in [
        1, 10, 32]] + [lods.shared_traincomplement]

    # Adding 2 lists, and checking on each shared_trainval or
    # shared_traincomplement alone
    for st in shared_trainvals + shared_traincomplements:
        assert_config_match(lods, st)
        print("Configuration Match Assertion OK")

    # Adding 2 datasets (one from shared_trainval and one from
    # shared_traincomplement and checking that configuraitons match across
    #  them both.
    for st1, st2 in zip(shared_trainvals, shared_traincomplements):
        joint_dataset = st1 + st2  # sum of 2 datasets
        assert_config_match(lods, joint_dataset)
        print("Configuration Match Assertion OK for joint_dataset ")


def main():
    # LODATAStruct_mul
    flat_test_workloads = [TEST_WORKLOADS[t] for t in TEST_WORKLOADS]
    flat_test_workloads = [e for l in flat_test_workloads for e in l]
    lods = make_LODatastruct_mul(
        flat_test_workloads, DATA_FOLDER, CONFIG_PATH, with_intensive=False,
        si=True, destroy=False)

    print("Running tests on newly created lods object....")
    check_shared_configs_consistency(lods)
    assert_no_shared_in_test(lods)
    assert_no_shared_in_test(lods, 'shared_traincomplement')

    lods.serialize(os.path.join(OUTPUT_FOLDER, "lods_mul_var.bin"),
                   destroy=True)

    # Now reads the serialized object and repeat the tests
    print("\nRepeating tests after reading serialized lods...")
    lods = None
    lods = LODataStruct.load_from_file(os.path.join(
        OUTPUT_FOLDER, "lods_mul_var.bin"), autobuild=True)

    check_shared_configs_consistency(lods)
    assert_no_shared_in_test(lods)
    assert_no_shared_in_test(lods, 'shared_traincomplement')


if __name__ == "__main__":
    main()
