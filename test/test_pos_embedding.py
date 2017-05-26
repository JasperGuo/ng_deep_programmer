# coding=utf8

"""
Test Position Embedding
"""

import tensorflow as tf

EMBEDDING_SIZE = 4
MAX_VALUE_SIZE = 5


def _calc_position_embedding(value_size, shape_0):
    """
    Calculate Position Embedding
    :param value_size: [shape_0]
    :return:
        [shape_0, max_value_size, digit_embedding_dim]
    """

    # Prevent divided by 0
    _value_size = tf.add(tf.cast(tf.equal(value_size, 0), dtype=tf.int32), value_size)

    # j/J
    j = tf.cast(tf.add(tf.range(MAX_VALUE_SIZE), 1), dtype=tf.float32)
    # Shape: [shape_0, max_value_size]
    replicated_j = tf.reshape(
        tf.tile(
            j,
            [shape_0]
        ),
        shape=[shape_0, MAX_VALUE_SIZE]
    )

    # Shape: [shape_0, 1]
    size_mask = tf.reshape(
        value_size,
        shape=[shape_0, 1]
    )
    # Shape: [shape_0, max_value_size]
    template = tf.reshape(
        tf.tile(
            tf.range(MAX_VALUE_SIZE),
            [shape_0]
        ),
        shape=[shape_0, MAX_VALUE_SIZE]
    )
    size_mask = tf.cast(
        tf.less(
            template, size_mask
        ),
        dtype=tf.float32
    )
    # Shape: [shape_0, max_value_size]
    position = tf.multiply(
        tf.div(
            replicated_j,
            tf.cast(
                tf.reshape(
                    _value_size,
                    shape=[shape_0, 1]
                ),
                dtype=tf.float32
            )
        ),
        size_mask
    )

    k_value = tf.cast(tf.add(tf.range(EMBEDDING_SIZE), 1), dtype=tf.float32)
    k_d_value = tf.div(k_value, tf.constant(EMBEDDING_SIZE, dtype=tf.float32))

    x = tf.subtract(
        tf.constant(1, dtype=tf.float32),
        k_d_value
    )

    result = tf.reshape(
        tf.subtract(
            x,
            tf.multiply(
                tf.reshape(position, shape=[shape_0 * MAX_VALUE_SIZE, 1]),
                tf.subtract(
                    tf.constant(1, dtype=tf.float32),
                    tf.multiply(
                        tf.constant(2, dtype=tf.float32),
                        k_d_value
                    )
                )
            )
        ),
        shape=[shape_0, MAX_VALUE_SIZE, EMBEDDING_SIZE]
    )

    result = tf.multiply(
        result,
        tf.reshape(
            size_mask,
            shape=[shape_0, MAX_VALUE_SIZE, 1]
        )
    )

    return result


if __name__ == "__main__":
    with tf.Session() as sess:
        value_size = tf.constant([0, 2, 3, 4, 5])
        position_Embedding = _calc_position_embedding(value_size, 5)
        print(position_Embedding.eval())
        print(position_Embedding.eval().shape)
