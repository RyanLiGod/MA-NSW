# -*- coding: utf-8 -*-

import random
import h5py

if __name__ == "__main__":
    pre_type = "glove"
    dimension = "200"
    data_type = ["base", "query"]
    hdf5_type = ["train", "test"]
    prefix = "../dataset/" + pre_type + "-" + dimension + "-angular.hdf5"
    ma_prefix = "../dataset/" + pre_type + dimension + "_ma/" + pre_type + dimension
    attr_1 = ["blue", "red", "green", "yellow"]
    attr_2 = ["sky", "land", "sea"]
    attr_3 = ["boy", "girl"]
    h5f = h5py.File(prefix, 'r')

    # Write base, query
    for i, t in enumerate(data_type):
        print("begin " + t)
        data = h5f[hdf5_type[i]]
        with open(ma_prefix + "_" + t + ".txt", 'w') as f:
            for line in data:
                for v in line:
                    f.write(str(v) + " ")
                f.write(attr_1[random.randint(0, 3)] + " ")
                f.write(attr_2[random.randint(0, 2)] + " ")
                f.write(attr_3[random.randint(0, 1)] + "\n")
