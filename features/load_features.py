#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-5-25 下午9:06
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import cPickle

# my own module
from conf.configure import Configure


def load_features():
    with open(Configure.processed_train_path, "rb") as f:
        train = cPickle.load(f)

    with open(Configure.processed_test_path, "rb") as f:
        test = cPickle.load(f)

    with open(Configure.processed_macro_path, "rb") as f:
        macro = cPickle.load(f)

    return train, test, macro
