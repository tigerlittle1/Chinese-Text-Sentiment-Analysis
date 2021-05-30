# -*- coding: utf-8 -*-
# @Time : 2021/4/6 23:01
# @Author : luff543
# @Email : luff543@gmail.com
# @File : metrics.py
# @Software: PyCharm

from engines.utils.extract_entity import extract_entity
import numpy as np
import tensorflow as tf


def metrics(y_true, y_pred):
    m = tf.keras.metrics.CategoricalAccuracy()
    m.update_state(list(y_true), list(y_pred))
    Accuracy = m.result()
#    recall = tf.keras.metrics.Recall(list(y_true), list(y_pred))
    return {"Accuracy":Accuracy}
