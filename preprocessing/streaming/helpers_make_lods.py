from sparkmodeling.common.lodatastruct import LODataStruct
import numpy as np
import os
from config import *


def make_LODatastruct_X(template_X, test_X, data_folder,
                        config_path,
                        with_intensive=False,
                        split_definitions=None,
                        X="X", destroy_on_serialize=False,
                        separate_intersections=False,
                        config_dict=None,
                        shared_within_templates=True):
    """ Makes and serializes LODS_X for left out template scenarios

    template_X: list
        List of workloads from the template X
    test_X: list
        List of test workloads from this template
    data_folder: str
        Path to folder containing csv files for each workloads data.
    config_path: str
        Path to a json configuration file that defines the splits.
    with_intensive: boolean
        Whether or not to include intensive workloads
    split_definitions: dict
        Contains split definitinons for the datasets trainval, traincomplement,
        test, shared_trainval, shared_traincomplement.
    config_dict: dict
        Dictionary containing the values of the global variables defined in
        config.py at the moment of creation of LODS.
    separate_intersections: boolean, default=False
        Whether or not to put the intersecting (shared) configurations across
        workloads into a separate dataset (prefixed by the word "shared")
    config_dict: dict
        Dictionary containing settings put into config.py when this LODS was created.
    shared_within_templates: boolean, default=True
        Whether the shared configurations intersections should be computed within templates
        or across all workloads (beyond template definition)
    """
    # X can be any of "A", "B", "C", ... "G" (templates)
    param_except_X = sorted(
        list(set(PARAM_WORKLOADS) - set(template_X)))

    aux = [TEST_WORKLOADS[t] for t in TEST_WORKLOADS]
    flatten_test_workloads = [e for l in aux for e in l]

    param_except_X_and_other_test = sorted(
        list(set(param_except_X) - set(flatten_test_workloads)))

    if separate_intersections:
        # jobs in template X but not in test (only used for intersections)
        # plus test jobs from other templates
        additional_jobs = list(set(template_X) - set(flatten_test_workloads))
        for temp_name in TEST_WORKLOADS:
            for j in TEST_WORKLOADS[temp_name]:
                if j not in test_X:
                    additional_jobs.append(j)
    else:
        additional_jobs = []

    if with_intensive:
        training_except_X = INTENSIVE_WORKLOADS + param_except_X_and_other_test
    else:
        training_except_X = param_except_X_and_other_test

    lods = LODataStruct(test_X, lo_config_path=config_path,
                        test_size=0.8)
    lods.read_jobs_data(training_except_X+test_X, data_folder,
                        additional_jobs=additional_jobs)
    if shared_within_templates:
        lods.build(
            imported_sd=split_definitions,
            separate_intersections=separate_intersections,
            templates=TEMPLATES)
    else:
        lods.build(
            imported_sd=split_definitions,
            separate_intersections=separate_intersections)

    if config_dict is not None:
        lods.set_config_dict(config_dict)
    lods.serialize(os.path.join(OUTPUT_FOLDER, "lods_{}.bin".format(X)),
                   destroy=destroy_on_serialize)


def make_LODatastruct_mul(
        test_workloads, data_folder, config_path, with_intensive=False,
        si=False, destroy_on_serialize=False, config_dict=None,
        shared_within_templates=True):
    """Makes and serializes LODS_mul for workload mapping scenarios.

    test_workloads: list
        Contains the list of test workloads
    data_folder: str
        Path to folder containing csv files for each workloads data.
    config_path: str
        Path to a json configuration file that defines the splits.
    with_intensive: bool
        Whether or not to include intensive workloads
    si: bool, default=False
        si stands for 'separate intersections'. Indicates whether or not we
        need to separate intersecting (shared) configurations across a
        particular template from non-intersecting configurations.
    config_dict: dict
        Dictionary containing settings put into config.py when this LODS was created.
    shared_within_templates: boolean, default=True
        Whether the shared configurations intersections should be computed within templates
        or across all workloads (beyond template definition)
    """
    param_training_workloads = sorted(
        list(set(PARAM_WORKLOADS) - set(test_workloads)))

    if with_intensive:
        training_workloads = INTENSIVE_WORKLOADS + param_training_workloads
    else:
        training_workloads = param_training_workloads

    lods = LODataStruct(test_workloads, lo_config_path=config_path,
                        test_size=0.8)
    lods.read_jobs_data(training_workloads+test_workloads, data_folder)
    if shared_within_templates:
        lods.build(
            separate_intersections=si, templates=TEMPLATES)
    else:
        lods.build(separate_intersections=si)
    if config_dict is not None:
        lods.set_config_dict(config_dict)
    lods.serialize(os.path.join(OUTPUT_FOLDER, "lods_mul.bin"),
                   destroy=destroy_on_serialize)
    return lods


def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    config_dict = get_config_dict()
    # LODATAStruct_mul
    flat_test_workloads = [TEST_WORKLOADS[t] for t in TEST_WORKLOADS]
    flat_test_workloads = [e for l in flat_test_workloads for e in l]

    lods = make_LODatastruct_mul(
        flat_test_workloads, DATA_FOLDER, CONFIG_PATH,
        with_intensive=WITH_INTENSIVE, si=SEPARATE_INTERSECTIONS_MUL)
    sd = lods.get_split_definitions()
    lods.describe()
    for temp in TEMPLATES:
        make_LODatastruct_X(
            TEMPLATES[temp],
            TEST_WORKLOADS[temp],
            DATA_FOLDER, CONFIG_PATH, with_intensive=WITH_INTENSIVE,
            split_definitions=sd, X=temp,
            separate_intersections=SEPARATE_INTERSECTIONS_X)


if __name__ == "__main__":
    main()
