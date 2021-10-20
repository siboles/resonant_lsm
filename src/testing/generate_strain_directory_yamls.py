import os
import sys
import yaml

ROOT_DIRECTORY = os.path.abspath(sys.argv[-1])
if not os.path.exists(ROOT_DIRECTORY):
    raise FileNotFoundError(ROOT_DIRECTORY)

basenames = ["gn_0p1_sn_0p0",
             "gn_0p1_sn_0p1",
             "gn_0p1_sn_0p2",
             "gn_0p2_sn_0p0",
             "gn_0p2_sn_0p1",
             "gn_0p2_sn_0p2"]

for i in range(5):
    for name in basenames:
        directory_file_name = "_".join([name, "{:03d}.yaml".format(i + 1)])
        reference_directory = os.path.join("_".join([name, "{:03d}".format(i + 1)]), "reference_results")
        deformed_directories = []
        for j in range(10):
            deformed_directories.append(
                os.path.join("_".join([name, "{:03d}".format(i + 1)]), "def_{:03d}_results".format(j + 1)))

        data = {"reference": reference_directory,
                "deformed": deformed_directories}
        with open(os.path.join(ROOT_DIRECTORY, directory_file_name), 'w') as f:
            yaml.dump(data, f)