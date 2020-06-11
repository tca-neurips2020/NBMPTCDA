from config import *

print("TEMPLATES:")
print(TEMPLATES)
print("---------------------------------------")

print("TEST WORKLOADS:")
print(TEST_WORKLOADS)
print("---------------------------------------")

print("PARAM WORKLOADS:")
print(PARAM_WORKLOADS)
print("---------------------------------------")

print("[len(TEMPLATES[key] for key in TEMPLATES], sum(...)")
lens = [len(TEMPLATES[key]) for key in TEMPLATES]
print(lens, sum(lens))

print("[len(TEST_WORKLOADS[key] for key in TEST_WORKLOADS], sum(...)")
lens2 = [len(TEST_WORKLOADS[key]) for key in TEST_WORKLOADS]
print(lens2, sum(lens2))


print("# of training workloads: {}".format(len(training_workloads_str)))
print("# of test workoads: {}".format(len(test_workloads_str)))
