# coding=utf8

import tensorflow as tf


def softmax_with_mask(tensor, mask):
    """
    Calculate Softmax with mask
    :param tensor: [shape1, shape2]
    :param mask:   [shape1, shape2]
    :return:
    """
    exp_tensor = tf.exp(tensor)
    masked_exp_tensor = tf.multiply(exp_tensor, mask)
    total = tf.reshape(
        tf.reduce_sum(masked_exp_tensor, axis=1),
        shape=[-1, 1]
    )
    return tf.div(masked_exp_tensor, total)


def get_last_relevant(output, length):
    """
    RNN Output
    :param output:  [shape_0, shape_1, shape_2]
    :param length:  [shape_0]
    :return:
        [shape_0, shape_2]
    """
    shape_2 = tf.shape(output)[2]
    slices = list()
    for idx, l in enumerate(tf.unstack(length)):
        last = tf.slice(output, begin=[idx, l - 1, 0], size=[1, 1, shape_2])
        slices.append(last)
    lasts = tf.concat(slices, 0)
    return lasts
