# coding=utf8

from .. import util
import numpy as np
import tensorflow as tf


class BasicModel:

    def __init__(
            self,
            data_type_vocab_manager,
            operation_vocab_manager,
            digit_vocab_manager,
            lambda_vocab_manager,
            opts,
            is_test=False
    ):
        self._data_type_vocab_manager = data_type_vocab_manager
        self._operation_vocab_manager = operation_vocab_manager
        self._digit_vocab_manager = digit_vocab_manager
        self._lambda_vocab_manager = lambda_vocab_manager

        self._is_test = is_test

        self._max_memory_size = util.get_value(opts, "max_memory_size")
        self._max_value_size = util.get_value(opts, "max_value_size")
        self._max_argument_num = util.get_value(opts, "max_argument_num")

        if self._is_test:
            self._batch_size = 1
        else:
            self._batch_size = util.get_value(opts, "batch_size")

        self._case_num = util.get_value(opts, "case_num")

        self._batch_with_case_size = self._batch_size * self._case_num

        self._batch_with_case_and_memory_size = self._batch_with_case_size * self._max_memory_size

        # Embedding Size
        self._data_type_embedding_dim = util.get_value(opts, "data_type_embedding_dim")
        self._operation_embedding_dim = util.get_value(opts, "operation_embedding_dim")
        self._lambda_embedding_dim = util.get_value(opts, "lambda_embedding_dim")
        self._digit_embedding_dim = util.get_value(opts, "digit_embedding_dim")

        # Memory Encoder (DNN), Size
        self._memory_encoder_dim = util.get_value(opts, "memory_encoder_dim")

        # Operation Selector Hidden Size
        self._operation_selector_dim = util.get_value(opts, "operation_selector_dim")

        # Argument Selector Hidden Size
        self._argument_selector_dim = util.get_value(opts, "argument_selector_dim")

    def _build_input_nodes(self):
        with tf.name_scope("model_placeholder"):
            self._memory_entry_data_type = tf.placeholder(tf.int32, [self._batch_with_case_and_memory_size], name="memory_entry_data_type")
            self._memory_entry_value = tf.placeholder(tf.int32, [self._batch_with_case_and_memory_size, self._max_value_size], name="memory_entry_value")
            self._memory_size = tf.placeholder(tf.int32, [self._batch_with_case_size], name="memory_size")
            self._output_data_type = tf.placeholder(tf.int32, [self._batch_with_case_size], name="output_data_type")
            self._output_value = tf.placeholder(tf.int32, [self._batch_with_case_size, self._max_value_size], name="output_value")

            if not self._is_test:
                self._operation = tf.placeholder(tf.int32, [self._batch_size], name="operation")
                self._arguments = tf.placeholder(tf.int32, [self._batch_size, self._max_argument_num], name="arguments")

    def _build_embedding_layer(self):
        with tf.variable_scope("data_type_embedding_layer"):
            pad_embedding = tf.get_variable(
                initializer=tf.zeros([1, self._data_type_embedding_dim]),
                name="data_type_pad_embedding",
                trainable=False
            )
            data_type_embedding = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._data_type_vocab_manager.vocab_len - 1, self._data_type_embedding_dim],
                    stddev=0.5
                ),
                name="data_type_embedding"
            )
            data_type_embedding = tf.concat(pad_embedding, data_type_embedding)

        with tf.variable_scope("operation_embedding_layer"):
            pad_embedding = tf.get_variable(
                initializer=tf.zeros([1, self._operation_embedding_dim]),
                name="operation_pad_embedding",
                trainable=False
            )
            operation_embedding = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._operation_vocab_manager.vocab_len - 1, self._operation_embedding_dim],
                    stddev=0.5
                ),
                name="operation_embedding"
            )
            operation_embedding = tf.concat(pad_embedding, operation_embedding)

        with tf.variable_scope("digit_embedding_layer"):
            pad_embedding = tf.get_variable(
                initializer=tf.zeros([1, self._digit_embedding_dim]),
                name="digit_pad_embedding",
                trainable=False
            )
            digit_embedding = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._digit_vocab_manager.vocab_len - 1, self._digit_embedding_dim],
                    stddev=0.5
                ),
                name="digit_embedding"
            )
            digit_embedding = tf.concat(pad_embedding, digit_embedding)

        with tf.variable_scope("lambda_embedding_layer"):
            pad_embedding = tf.get_variable(
                initializer=tf.zeros([1, self._lambda_embedding_dim]),
                name="digit_pad_embedding",
                trainable=False
            )
            lambda_embedding = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._digit_vocab_manager.vocab_len - 1, self._lambda_embedding_dim],
                    stddev=0.5
                ),
                name="lambda_embedding"
            )
            lambda_embedding = tf.concat(pad_embedding, lambda_embedding)

        return data_type_embedding, operation_embedding, digit_embedding, lambda_embedding

    def _build_train_graph(self):
        self._build_input_nodes()

        data_type_embedding, operation_embedding, digit_embedding, lambda_embedding = self._build_embedding_layer()

    def _build_test_graph(self):
        pass
