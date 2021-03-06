# -*- coding: utf-8 -*-

import random

import numpy as np


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


if __name__ == "__main__":
    pre_type = "sift"
    percent = "1_16"
    data_type = ["base", "query"]
    prefix = "../dataset/" + pre_type + "/" + pre_type
    ma_prefix = "../dataset/" + pre_type + percent + "_ma/" + pre_type + percent
    attr_1 = ["blue", "red", "green", "yellow"]
    attr_2 = ["sky", "land", "sea"]
    attr_3 = ["boy", "girl", "girl", "girl", "girl", "girl", "girl", "girl", "girl", "girl", "girl", "girl", "girl",
              "girl", "girl", "girl"]
    data = []
    for t in data_type:
        data.append(fvecs_read(prefix + "_" + t + ".fvecs"))

    # Write base, query, learn
    for i, t in enumerate(data_type):
        print("begin " + t)
        with open(ma_prefix + "_" + t + ".txt", 'w') as f:
            for line in data[i]:
                for v in line:
                    f.write(str(v) + " ")
                f.write(attr_1[random.randint(0, 3)] + " ")
                f.write(attr_2[random.randint(0, 2)] + " ")
                f.write(attr_3[random.randint(0, 15)] + "\n")
