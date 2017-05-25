# coding=utf8

import sys

sys.path.insert(0, "..")

import tensorflow as tf
import util
import models_util


class FFContextModel:
    epsilon = 1e-5

    def __init__(
            self,
            data_type_vocab_manager,
            operation_vocab_manager,
            digit_vocab_manager,
            opts,
            is_test=False
    ):
        self._data_type_vocab_manager = data_type_vocab_manager
        self._operation_vocab_manager = operation_vocab_manager
        self._digit_vocab_manager = digit_vocab_manager

        self._is_test = is_test

        self._max_memory_size = util.get_value(opts, "max_memory_size")
        self._max_value_size = util.get_value(opts, "max_value_size")
        self._max_argument_num = util.get_value(opts, "max_argument_num")

        # if self._is_test:
        #     self._batch_size = 1
        # else:
        self._batch_size = util.get_value(opts, "batch_size")

        self._case_num = util.get_value(opts, "case_num")

        self._batch_with_case_size = self._batch_size * self._case_num

        self._batch_with_case_and_memory_size = self._batch_with_case_size * self._max_memory_size

        self._dropout = util.get_value(opts, "dropout", 0.25)

        # Embedding Size
        self._data_type_embedding_dim = util.get_value(opts, "data_type_embedding_dim")
        self._operation_embedding_dim = util.get_value(opts, "operation_embedding_dim")
        self._lambda_embedding_dim = util.get_value(opts, "lambda_embedding_dim")
        self._digit_embedding_dim = util.get_value(opts, "digit_embedding_dim")

        # Memory Encoder (DNN), Size
        self._memory_encoder_layer_1_dim = util.get_value(opts, "memory_encoder_layer_1_dim")
        self._memory_encoder_layer_2_dim = util.get_value(opts, "memory_encoder_layer_2_dim")

        # Output Encoder (DNN), Size
        self._output_encoder_layer_1_dim = util.get_value(opts, "output_encoder_layer_1_dim")
        self._output_encoder_layer_2_dim = util.get_value(opts, "output_encoder_layer_2_dim")

        # Guide Hidden Layer
        self._guide_hidden_dim = util.get_value(opts, "guide_hidden_dim")

        # Operation Selector Hidden Size
        self._operation_selector_dim = util.get_value(opts, "operation_selector_dim")

        self._gradient_clip = util.get_value(opts, "gradient_clip", 5)

        if self._is_test:
            self._build_test_graph()
        else:
            self._build_train_graph()

    def _build_input_nodes(self):
        with tf.name_scope("model_placeholder"):
            self._memory_entry_data_type = tf.placeholder(tf.int32, [self._batch_with_case_and_memory_size],
                                                          name="memory_entry_data_type")
            self._memory_entry_value = tf.placeholder(tf.int32,
                                                      [self._batch_with_case_and_memory_size, self._max_value_size],
                                                      name="memory_entry_value")
            self._memory_entry_value_size = tf.placeholder(tf.int32,
                                                           [self._batch_with_case_and_memory_size],
                                                           name="memory_entry_value_size")
            self._memory_entry_scr_1 = tf.placeholder(tf.int32, [
                self._batch_with_case_and_memory_size
            ], name="memory_entry_scr_1")
            self._memory_entry_scr_2 = tf.placeholder(tf.int32, [
                self._batch_with_case_and_memory_size
            ], name="memory_entry_scr_2")
            self._memory_entry_operation = tf.placeholder(tf.int32, [
                self._batch_with_case_and_memory_size
            ], name="memory_entry_operation")
            self._memory_size = tf.placeholder(tf.int32, [self._batch_with_case_size], name="memory_size")
            self._output_data_type = tf.placeholder(tf.int32, [self._batch_with_case_size], name="output_data_type")
            self._output_value = tf.placeholder(tf.int32, [self._batch_with_case_size, self._max_value_size],
                                                name="output_value")
            self._output_value_size = tf.placeholder(tf.int32, [self._batch_with_case_size], name="output_value_size")
            self._dnn_keep_prob = tf.placeholder(tf.float32, name="dnn_keep_prob")

            if not self._is_test:
                self._operation = tf.placeholder(tf.int32, [self._batch_size], name="operation")
                self._learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    def _build_embedding_layer(self):
        with tf.variable_scope("data_type_embedding_layer"):
            pad_embedding = tf.get_variable(
                initializer=tf.zeros([1, self._data_type_embedding_dim], dtype=tf.float32),
                name="data_type_pad_embedding",
                trainable=False
            )
            data_type_embedding = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._data_type_vocab_manager.vocab_len - 1, self._data_type_embedding_dim],
                    stddev=0.5
                ),
                name="data_type_embedding",
                dtype=tf.float32,
            )
            data_type_embedding = tf.concat([pad_embedding, data_type_embedding], axis=0)

        with tf.variable_scope("digit_embedding_layer"):
            pad_embedding = tf.get_variable(
                initializer=tf.zeros([1, self._digit_embedding_dim], dtype=tf.float32),
                name="digit_pad_embedding",
                trainable=False
            )
            digit_embedding = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._digit_vocab_manager.vocab_len - 1, self._digit_embedding_dim],
                    stddev=0.5
                ),
                dtype=tf.float32,
                name="digit_embedding"
            )
            digit_embedding = tf.concat([pad_embedding, digit_embedding], axis=0)

        with tf.variable_scope("operation_embedding_layer"):
            pad_embedding = tf.get_variable(
                initializer=tf.zeros([1, self._operation_embedding_dim], dtype=tf.float32),
                name="operation_pad_embedding",
                trainable=False
            )
            operation_embedding = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._operation_vocab_manager.vocab_len - 1, self._operation_embedding_dim],
                    stddev=0.5
                ),
                dtype=tf.float32,
                name="operation_embedding"
            )
            operation_embedding = tf.concat([pad_embedding, operation_embedding], axis=0)

        return data_type_embedding, digit_embedding, operation_embedding

    def _calc_position_embedding(self, value_size, shape_0):
        """
        Calculate Position Embedding
        :param value_size: [shape_0]
        :return:
            [shape_0, max_value_size, digit_embedding_dim]
        """

        # Prevent divided by 0
        _value_size = tf.add(tf.cast(tf.equal(value_size, 0), dtype=tf.int32), value_size)

        # j/J
        j = tf.cast(tf.add(tf.range(self._max_value_size), 1), dtype=tf.float32)
        # Shape: [shape_0, max_value_size]
        replicated_j = tf.reshape(
            tf.tile(
                j,
                [shape_0]
            ),
            shape=[shape_0, self._max_value_size]
        )

        # Shape: [shape_0, 1]
        size_mask = tf.reshape(
            value_size,
            shape=[shape_0, 1]
        )
        # Shape: [shape_0, max_value_size]
        template = tf.reshape(
            tf.tile(
                tf.range(self._max_value_size),
                [shape_0]
            ),
            shape=[shape_0, self._max_value_size]
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

        k_value = tf.cast(tf.add(tf.range(self._digit_embedding_dim), 1), dtype=tf.float32)
        k_d_value = tf.div(k_value, tf.constant(self._digit_embedding_dim, dtype=tf.float32))

        x = tf.subtract(
            tf.constant(1, dtype=tf.float32),
            k_d_value
        )

        result = tf.reshape(
            tf.subtract(
                x,
                tf.multiply(
                    tf.reshape(position, shape=[shape_0 * self._max_value_size, 1]),
                    tf.subtract(
                        tf.constant(1, dtype=tf.float32),
                        tf.multiply(
                            tf.constant(2, dtype=tf.float32),
                            k_d_value
                        )
                    )
                )
            ),
            shape=[shape_0, self._max_value_size, self._digit_embedding_dim]
        )

        result = tf.multiply(
            result,
            tf.reshape(
                size_mask,
                shape=[shape_0, self._max_value_size, 1]
            )
        )

        return result

    def _build_memory_encoder(self):
        with tf.variable_scope("memory_encoder"):
            layer_1_weights = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._data_type_embedding_dim + self._digit_embedding_dim * self._max_value_size,
                       self._memory_encoder_layer_1_dim],
                name="weights_1"
            )
            layer_1_bias = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[self._memory_encoder_layer_1_dim],
                name="bias_1"
            )

            layer_2_weights = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._memory_encoder_layer_1_dim,
                       self._memory_encoder_layer_2_dim],
                name="weights_2"
            )
            layer_2_bias = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[self._memory_encoder_layer_2_dim],
                name="bias_2"
            )

            weights = {
                "W1": layer_1_weights,
                "W2": layer_2_weights
            }

            bias = {
                "b1": layer_1_bias,
                "b2": layer_2_bias
            }

            return weights, bias

    def _build_output_encoder(self):
        with tf.variable_scope("output_encoder"):
            layer_1_weights = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._output_encoder_layer_2_dim,
                       self._output_encoder_layer_2_dim],
                name="weights_1"
            )
            layer_1_bias = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[self._output_encoder_layer_2_dim],
                name="bias_1"
            )
            """
            layer_2_weights = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._output_encoder_layer_1_dim,
                       self._output_encoder_layer_2_dim],
                name="weights_2"
            )
            layer_2_bias = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[self._output_encoder_layer_2_dim],
                name="bias_2"
            )
            """
            weights = {
                "W1": layer_1_weights,
                # "W2": layer_2_weights
            }

            bias = {
                "b1": layer_1_bias,
                # "b2": layer_2_bias
            }

            return weights, bias

    def _encode_memory(self, weights, biases, data_type_embedded, value_embedded, value_size):
        """
        Encode memory entry
        :param weights:
        :param biases:
        :param data_type_embedded: [batch_size*case_num*max_memory_size, data_type_embedding_dim]
        :param value_embedded:     [batch_size*case_num*max_memory_size, max_value_size, digit_embedding_dim]
        :param value_size:         [batch_size*case_num*max_memory_size]
        :return:
            [batch_size*case_num*max_memory_size, memory_encoder_layer_2_dim]
        """
        # Shape:  [batch_size*case_num*max_memory_size, max_value_size, digit_embedding_dim]
        position_embedding = self._calc_position_embedding(value_size, shape_0=self._batch_with_case_and_memory_size)
        position_augmented_value_embedding = tf.add(
            value_embedded, position_embedding
        )

        # Shape: [batch_size*case_max_memory_size, max_value_size*digit_embedding+data_type_embedding_dim]
        concatenated_memory_entry_embedded = tf.concat(
            (
                tf.reshape(position_augmented_value_embedding, shape=[self._batch_with_case_and_memory_size,
                                                                      self._max_value_size * self._digit_embedding_dim]),
                data_type_embedded
            ),
            axis=1
        )

        with tf.name_scope("encode_memory"):
            layer_1 = tf.add(tf.matmul(concatenated_memory_entry_embedded, weights["W1"]), biases["b1"])
            layer_1 = tf.nn.relu(layer_1)
            layer_1 = tf.nn.dropout(layer_1, self._dnn_keep_prob)
            layer_2 = tf.add(tf.matmul(layer_1, weights["W2"]), biases["b2"])

            return layer_2

    def _encode_output(self, weights, biases, projection_weights, projection_bias, data_type_embedded, value_embedded,
                       value_size):
        """
        Encode output
        :param weights:
        :param biases:
        :param data_type_embedded: [batch_size*case_num, data_type_embedding_dim]
        :param value_embedded:     [batch_size*case_num, max_value_size, data_type_embedding_dim]
        :param value_size:         [batch_size*case_num*max_memory_size]
        :return:
            [batch_size*case_num, output_encoder_layer_2_dim]
        """
        # Shape:  [batch_size*case_num, max_value_size, digit_embedding_dim]
        position_embedding = self._calc_position_embedding(value_size, shape_0=self._batch_with_case_size)
        position_augmented_value_embedding = tf.add(
            value_embedded, position_embedding
        )
        # Shape: [batch_size*case_num, max_value_size*data_type_embedding_dim]
        concatenated_output_embedded = tf.concat(
            (
                tf.reshape(position_augmented_value_embedding,
                           shape=[self._batch_with_case_size, self._max_value_size * self._digit_embedding_dim]),
                data_type_embedded
            ),
            axis=1
        )

        with tf.name_scope("encode_output"):
            layer_1 = tf.add(tf.matmul(concatenated_output_embedded, weights["W1"]), biases["b1"])
            layer_1 = tf.nn.relu(layer_1)
            layer_1 = tf.nn.dropout(layer_1, self._dnn_keep_prob)
            layer_2 = tf.add(tf.matmul(layer_1, weights["W2"]), biases["b2"])

            layer_3 = tf.tanh(
                tf.add(
                    tf.matmul(
                        layer_2,
                        projection_weights["W1"]
                    ),
                    projection_bias["b1"]
                )
            )

            return layer_3

    def _build_context_encoder(self):
        with tf.variable_scope("context_weights"):
            operation_weights = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._operation_embedding_dim,
                       self._memory_encoder_layer_2_dim],
                name="operation_weights"
            )

            source_1_weights = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._memory_encoder_layer_2_dim,
                       self._memory_encoder_layer_2_dim],
                name="source_1_weights"
            )

            source_2_weights = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._memory_encoder_layer_2_dim,
                       self._memory_encoder_layer_2_dim],
                name="source_2_weights"
            )

            bias = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[self._memory_encoder_layer_2_dim],
                name="bias"
            )

            layer_2_weights = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._memory_encoder_layer_2_dim,
                       self._memory_encoder_layer_2_dim],
                name="layer_2_weights"
            )

            layer_2_bias = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[self._memory_encoder_layer_2_dim],
                name="layer_2_bias"
            )

            combine_value_weights_1 = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._memory_encoder_layer_2_dim,
                       self._memory_encoder_layer_2_dim],
                name="combine_value_weights_1"
            )

            combine_value_weights_2 = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._memory_encoder_layer_2_dim,
                       self._memory_encoder_layer_2_dim],
                name="combine_value_weights_2"
            )

            combine_value_bias = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[self._memory_encoder_layer_2_dim],
                name="combine_value_bias"
            )

            w = {
                "operation_weights": operation_weights,
                "source_1_weights": source_1_weights,
                "source_2_weights": source_2_weights,
                "layer_2_weights": layer_2_weights,
                "combine_value_weights_1": combine_value_weights_1,
                "combine_value_weights_2": combine_value_weights_2
            }

            b = {
                "bias": bias,
                "layer_2_bias": layer_2_bias,
                "combine_value_bias": combine_value_bias
            }

            return w, b

    def _encode_context(self, weights, bias, encoded_memory, memory_entry_src1, memory_entry_src2, memory_embedded_opt):
        """
        Encode Context
        :param weights:
                {
                    "operation_weights":,
                    "source_1_weights":,
                    "source_2_weights
                }
        :param bias:
                {
                    "bias":
                }
        :param encoded_memory:      [batch_size*case_num*max_memory_size, memory_encoder_layer_2_dim]
        :param memory_entry_src1:   [batch_size*case_num*max_memory_size]
        :param memory_entry_src2:   [batch_size*case_num*max_memory_size]
        :param memory_embedded_opt:          [batch_size*case_num*max_memory_size, operation_embedding_dim]
        :return:
            [batch_size*case_num*max_memory_size, operation_embedding_dim]
        """
        memory_size = self._max_memory_size + 1
        # Expand Memory
        zeros = tf.zeros([self._batch_with_case_size, 1, self._memory_encoder_layer_2_dim], dtype=tf.float32)
        # Shape: [batch_size*case_num, max_memory_size+1, memory_encoder_layer_2_dim]
        expaned_memory = tf.concat(
            [
                tf.reshape(
                    encoded_memory,
                    shape=[self._batch_with_case_size, self._max_memory_size, self._memory_encoder_layer_2_dim]
                ),
                zeros
            ],
            axis=1
        )

        # Calculate index
        first_index = tf.reshape(
            tf.tile(
                tf.reshape(
                    tf.range(self._batch_with_case_size),
                    shape=[self._batch_with_case_size, 1]
                ),
                [1, self._max_memory_size]
            ),
            shape=[self._batch_with_case_and_memory_size]
        )
        # Shape: [batch_size*case_num*max_memory_size, batch_size*case_num*max_memory_size]
        src1_indices = tf.stack(
            [
                first_index,
                memory_entry_src1
            ],
            axis=1
        )
        # Shape: [batch_size*case_num*max_memory_size, batch_size*case_num*max_memory_size]
        src2_indices = tf.stack(
            [
                first_index,
                memory_entry_src2
            ],
            axis=1
        )

        # Shape: [batch_size*case_num*max_memory_size, memory_encoder_layer_2_dim]
        source_1 = tf.gather_nd(expaned_memory, src1_indices)
        source_2 = tf.gather_nd(expaned_memory, src2_indices)

        # Shape: [batch_size*case_num*max_memory_size, memory_encoder_layer_2_dim]

        layer_1 = tf.nn.relu(
            tf.add(
                tf.add_n(
                    [
                        tf.matmul(
                            source_1,
                            weights["source_1_weights"]
                        ),
                        tf.matmul(
                            source_2,
                            weights["source_2_weights"]
                        ),
                        tf.matmul(
                            memory_embedded_opt,
                            weights["operation_weights"]
                        )
                    ]
                ),
                bias["bias"]
            )
        )
        layer_1 = tf.nn.relu(layer_1)
        layer_1 = tf.nn.dropout(layer_1, self._dnn_keep_prob)

        # Shape: [batch_size*case_num*max_memory_size, memory_encoder_layer_2_dim]
        layer_2 = tf.add(tf.matmul(layer_1, weights["layer_2_weights"]), bias["layer_2_bias"])

        # Shape: [batch_size*case_num*max_memory_size, memory_encoder_layer_2_dim]
        combined = tf.nn.relu(
            tf.add(
                tf.add_n([
                    tf.matmul(
                        layer_2,
                        weights["combine_value_weights_1"]
                    ),
                    tf.matmul(
                        encoded_memory,
                        weights["combine_value_weights_2"]
                    )
                ]),
                bias["combine_value_bias"]
            )
        )

        return combined

    def _build_memory_output_attention_layer(self):
        with tf.variable_scope("memory_output_attention"):
            weights = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._memory_encoder_layer_2_dim,
                       self._memory_encoder_layer_2_dim],
                name="weights"
            )

            bias = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[self._memory_encoder_layer_2_dim],
                name="bias"
            )
            score_weights_1 = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._memory_encoder_layer_2_dim,
                       self._output_encoder_layer_2_dim],
                name="score_weights_1"
            )

            w = {
                "context_activate_weights": weights,
                "score_weights_1": score_weights_1
            }

            b = {
                "context_activate_bias": bias
            }

            return w, b

    def _calc_attention_scores(self, score_weights, encoded_memory, encoded_output):
        """
        Calculate attention score
        :param score_weights:
        :param encoded_memory: [batch_size*case_num*max_memory_size, memory_encoder_layer_2_dim]
        :param encoded_output: [batch_size*case_num*max_memory_size, output_encoder_layer_2_dim]
        :return:
            [batch_size*case_num*max_memory_size]
        """

        scores = tf.reduce_sum(
            tf.multiply(
                tf.matmul(encoded_memory, score_weights["score_weights_1"]),
                encoded_output
            ),
            axis=1
        )
        return scores

    def _calc_memory_output_attention(self, attention_weights, attention_bias, encoded_memory,
                                      encoded_output, memory_size):
        """
        Calculate Memory, Output Attention
        :param encoded_memory:      [batch_size*case_num*memory_size, memory_encoder_layer_2_dim]
        :param encoded_output:      [batch_size*case_num, output_encoder_layer_2_dim]
        :param memory_size:         [batch_size*case_num]
        :return:
            [batch_size*case_num, memory_encoder_layer_2_dim]
        """
        # Replicate encoded_output
        # Shape: [batch_size*case_num*max_memory_size, output_encoder_layer_2_dim]
        replicated_encoded_output = tf.reshape(
            tf.reshape(
                tf.tile(encoded_output, [1, self._max_memory_size]),
                shape=[self._batch_with_case_size, self._max_memory_size, self._output_encoder_layer_2_dim]
            ),
            shape=[self._batch_with_case_and_memory_size, self._output_encoder_layer_2_dim]
        )

        # Calculate Attention Score
        # Shape: [batch_size*case_num, max_memory_size]
        scores = tf.reshape(
            self._calc_attention_scores(attention_weights, encoded_memory, replicated_encoded_output),
            shape=[self._batch_with_case_size, self._max_memory_size]
        )

        # Mask
        memory_mask = tf.reshape(
            memory_size,
            shape=[self._batch_with_case_size, 1]
        )
        memory_template = tf.reshape(
            tf.tile(
                tf.range(self._max_memory_size),
                [self._batch_with_case_size]
            ),
            shape=[self._batch_with_case_size, self._max_memory_size]
        )
        memory_mask = tf.cast(
            tf.less(
                memory_template, memory_mask
            ),
            dtype=tf.float32
        )

        # Shape: [batch_size*case_num, max_memory_size]
        normalized_weights = models_util.softmax_with_mask(scores, memory_mask)
        reshaped_encoded_memory = tf.reshape(
            encoded_memory,
            shape=[self._batch_with_case_size, self._max_memory_size, self._memory_encoder_layer_2_dim]
        )
        # Shape: [batch_size*case_num, memory_encoder_layer_2_dim]
        context_vector = tf.reduce_sum(
            tf.multiply(
                tf.reshape(
                    normalized_weights, shape=[self._batch_with_case_size, self._max_memory_size, 1]
                ),
                reshaped_encoded_memory
            ),
            axis=1
        )

        attentive_context_vector = tf.tanh(
            tf.add(
                tf.matmul(
                    context_vector, attention_weights["context_activate_weights"], transpose_b=True
                ),
                attention_bias["context_activate_bias"]
            )
        )

        return attentive_context_vector, normalized_weights

    def _build_guide_layer(self):
        with tf.variable_scope("guide"):
            attentive_context_weights = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._guide_hidden_dim,
                       self._memory_encoder_layer_2_dim],
                name="attentive_context_weights"
            )
            output_weights = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._guide_hidden_dim,
                       self._output_encoder_layer_2_dim],
                name="output_weights"
            )
            bias = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[self._guide_hidden_dim],
                name="bias"
            )

            return attentive_context_weights, output_weights, bias

    def _calc_guide_vector(self, guide_context_weights, guide_output_weights, guide_bias, context_vector,
                           encoded_output):
        """
            RELU(W1*context_vector + W2*encoded_output + bias)
        :param guide_context_weights:   [guide_hidden_dim, memory_encoder_layer_2_dim*2]
        :param guide_output_weights:    [guide_hidden_dim, output_encoder_layer_2_dim]
        :param guide_bias:              [guide_hidden_dim]
        :param context_vector:          [batch_size*case_num, memory_encoder_layer_2_dim*2]
        :param encoded_output:          [batch_size*case_num, output_encoder_layer_2_dim]
        :return:
            [batch_size*case_num, guide_hidden_Dim]
        """
        return tf.nn.relu(
            tf.add(
                tf.add(
                    tf.matmul(
                        context_vector,
                        guide_context_weights,
                        transpose_b=True
                    ),
                    tf.matmul(
                        encoded_output,
                        guide_output_weights,
                        transpose_b=True
                    )
                ),
                guide_bias
            )
        )

    def _build_operation_selector(self):
        with tf.variable_scope("operation_selector"):
            layer_1_weights = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._guide_hidden_dim,
                       self._operation_selector_dim],
                name="weights_1"
            )
            layer_1_bias = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[self._operation_selector_dim],
                name="bias_1"
            )

            output_weights = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._operation_selector_dim,
                       self._operation_embedding_dim],
                name="output_weights"
            )
            output_bias = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[self._operation_embedding_dim],
                name="output_bias"
            )

            weights = {
                "W1": layer_1_weights,
                "output_W": output_weights
            }

            bias = {
                "b1": layer_1_bias,
                "output_b": output_bias
            }

            return weights, bias

    def _select_operation(self, selector_weights, selector_biases, guide_vector, operation_embedding):
        """
        :param selector_weights:
        :param selector_biases:
        :param guide_vector: [batch_size*case_num, guide_hidden_dim]
        :param operation_embedding: [operation_vocab_len, operation_embedding_dim]
        :return:
            [batch_size, operation_vocab_len], [batch_size]
        """
        with tf.name_scope("select_operation"):
            # Shape: [batch_size*case_num, operation_selector_dim]
            layer_1 = tf.add(tf.matmul(guide_vector, selector_weights["W1"]), selector_biases["b1"])
            layer_1 = tf.nn.relu(layer_1)
            output_layer = tf.add(tf.matmul(layer_1, selector_weights["output_W"]),
                                  selector_biases["output_b"])

            # Shape: [batch_size, operation_embedding_dim]
            pooled_output_layer = tf.reduce_sum(
                tf.transpose(
                    tf.reshape(
                        output_layer,
                        shape=[self._batch_size, self._case_num, self._operation_embedding_dim]
                    ),
                    perm=[0, 2, 1]
                ),
                axis=2
            )

            with tf.name_scope("calc_operation_attention"):
                # Calculate attention to operation embedding

                # Shape: [batch_size, operation_vocab_len]
                attention_scores = tf.matmul(
                    pooled_output_layer,
                    operation_embedding,
                    transpose_b=True
                )

                softmax_output = tf.nn.softmax(attention_scores, dim=-1)
                selection = tf.arg_max(softmax_output, dimension=1)

                return softmax_output, selection

    def _build_train_graph(self):
        """
        Build Training Graph
        :return:
        """
        self._build_input_nodes()

        data_type_embedding, digit_embedding, operation_embedding = self._build_embedding_layer()

        memory_entry_data_type_embedded = tf.nn.embedding_lookup(data_type_embedding, self._memory_entry_data_type)
        memory_entry_value_embedded = tf.nn.embedding_lookup(digit_embedding, self._memory_entry_value)
        output_data_type_embedded = tf.nn.embedding_lookup(data_type_embedding, self._output_data_type)
        output_value_embedded = tf.nn.embedding_lookup(digit_embedding, self._output_value)
        operation_embedded = tf.nn.embedding_lookup(operation_embedding, self._memory_entry_operation)

        memory_encoder_weights, memory_encoder_biases = self._build_memory_encoder()
        output_encoder_weights, output_encoder_biases = self._build_output_encoder()

        # Encode Memory and Output
        encoded_memory = self._encode_memory(memory_encoder_weights, memory_encoder_biases,
                                             memory_entry_data_type_embedded, memory_entry_value_embedded,
                                             self._memory_entry_value_size)
        encoded_output = self._encode_output(memory_encoder_weights,
                                             memory_encoder_biases,
                                             output_encoder_weights,
                                             output_encoder_biases,
                                             output_data_type_embedded,
                                             output_value_embedded,
                                             self._output_value_size)

        context_encoder_weights, context_encoder_bias = self._build_context_encoder()
        context_encoded_memory = self._encode_context(
            weights=context_encoder_weights,
            bias=context_encoder_bias,
            encoded_memory=encoded_memory,
            memory_entry_src1=self._memory_entry_scr_1,
            memory_entry_src2=self._memory_entry_scr_2,
            memory_embedded_opt=operation_embedded
        )
        # Guide
        attention_weights, attention_bias = self._build_memory_output_attention_layer()
        attentive_context_vector, self._attentive_context_weights = self._calc_memory_output_attention(
            attention_weights=attention_weights,
            attention_bias=attention_bias,
            encoded_memory=context_encoded_memory,
            encoded_output=encoded_output,
            memory_size=self._memory_size
        )

        guide_context_weights, guide_output_weights, guide_bias = self._build_guide_layer()
        # Shape: [batch_size*case_num, guide_hidden_dim]
        guide_vector = self._calc_guide_vector(
            guide_context_weights=guide_context_weights,
            guide_output_weights=guide_output_weights,
            guide_bias=guide_bias,
            context_vector=attentive_context_vector,
            encoded_output=encoded_output
        )

        # Operation Selector
        operation_selector_weights, operation_selector_biases = self._build_operation_selector()

        # Shape: [batch_size, operation_vocab_len], [batch_size]
        operation_softmax_output, selected_operations = self._select_operation(
            selector_weights=operation_selector_weights,
            selector_biases=operation_selector_biases,
            guide_vector=guide_vector,
            operation_embedding=operation_embedding
        )

        # Prevent inf
        operation_softmax_output = tf.add(operation_softmax_output, tf.constant(self.epsilon, dtype=tf.float32))

        self._operation_prediction = selected_operations

        with tf.name_scope("loss"):
            #############################################################################
            # Operation Probs
            # Calculate Operation Index
            prefix_index = tf.expand_dims(tf.range(self._batch_size), dim=1)
            truth_operation_indices = tf.concat(
                [
                    prefix_index,
                    tf.reshape(self._operation, shape=[self._batch_size, 1])
                ],
                axis=1
            )
            # Shape: [batch_size]
            # To calculate loss
            truth_operations_probs = tf.gather_nd(operation_softmax_output, truth_operation_indices)

            # [batch_size]
            log_probs = tf.log(truth_operations_probs)

            vars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars
                                if 'bias' not in v.name]) * 0.001
            self._loss = tf.add(
                tf.negative(
                    tf.reduce_mean(log_probs),
                ),
                l2_loss
            )

        with tf.name_scope("back_propagation"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
            # self._optimizer = optimizer.minimize(self._loss)

            # clipped at 5 to alleviate the exploding gradient problem
            self._gvs = optimizer.compute_gradients(self._loss)
            self._capped_gvs = [(tf.clip_by_value(grad, -self._gradient_clip, self._gradient_clip), var) for grad, var
                                in self._gvs]
            self._optimizer = optimizer.apply_gradients(self._capped_gvs)

    def _build_test_graph(self):
        """
        Build Test Graph
        :return:
        """
        self._build_input_nodes()

        data_type_embedding, digit_embedding, operation_embedding = self._build_embedding_layer()

        memory_entry_data_type_embedded = tf.nn.embedding_lookup(data_type_embedding, self._memory_entry_data_type)
        memory_entry_value_embedded = tf.nn.embedding_lookup(digit_embedding, self._memory_entry_value)
        output_data_type_embedded = tf.nn.embedding_lookup(data_type_embedding, self._output_data_type)
        output_value_embedded = tf.nn.embedding_lookup(digit_embedding, self._output_value)
        operation_embedded = tf.nn.embedding_lookup(operation_embedding, self._memory_entry_operation)

        memory_encoder_weights, memory_encoder_biases = self._build_memory_encoder()
        output_encoder_weights, output_encoder_biases = self._build_output_encoder()

        # Encode Memory and Output
        encoded_memory = self._encode_memory(memory_encoder_weights, memory_encoder_biases,
                                             memory_entry_data_type_embedded, memory_entry_value_embedded,
                                             self._memory_entry_value_size)
        encoded_output = self._encode_output(memory_encoder_weights,
                                             memory_encoder_biases,
                                             output_encoder_weights,
                                             output_encoder_biases,
                                             output_data_type_embedded,
                                             output_value_embedded,
                                             self._output_value_size)

        context_encoder_weights, context_encoder_bias = self._build_context_encoder()
        context_encoded_memory = self._encode_context(
            weights=context_encoder_weights,
            bias=context_encoder_bias,
            encoded_memory=encoded_memory,
            memory_entry_src1=self._memory_entry_scr_1,
            memory_entry_src2=self._memory_entry_scr_2,
            memory_embedded_opt=operation_embedded
        )
        # Guide
        attention_weights, attention_bias = self._build_memory_output_attention_layer()
        attentive_context_vector, self._attentive_context_weights = self._calc_memory_output_attention(
            attention_weights=attention_weights,
            attention_bias=attention_bias,
            encoded_memory=context_encoded_memory,
            encoded_output=encoded_output,
            memory_size=self._memory_size
        )

        guide_context_weights, guide_output_weights, guide_bias = self._build_guide_layer()
        # Shape: [batch_size*case_num, guide_hidden_dim]
        guide_vector = self._calc_guide_vector(
            guide_context_weights=guide_context_weights,
            guide_output_weights=guide_output_weights,
            guide_bias=guide_bias,
            context_vector=attentive_context_vector,
            encoded_output=encoded_output
        )

        # Operation Selector
        operation_selector_weights, operation_selector_biases = self._build_operation_selector()

        # Shape: [batch_size, operation_vocab_len], [batch_size]
        operation_softmax_output, selected_operations = self._select_operation(
            selector_weights=operation_selector_weights,
            selector_biases=operation_selector_biases,
            guide_vector=guide_vector,
            operation_embedding=operation_embedding
        )
        self._operation_prediction = selected_operations

    def _build_train_feed(self, batch):
        feed_dict = dict()
        feed_dict[self._memory_entry_data_type] = batch.memory_entry_data_type
        feed_dict[self._memory_entry_value] = batch.memory_entry_value
        feed_dict[self._memory_entry_value_size] = batch.memory_entry_value_size
        feed_dict[self._memory_entry_scr_1] = batch.memory_entry_src1
        feed_dict[self._memory_entry_scr_2] = batch.memory_entry_src2
        feed_dict[self._memory_entry_operation] = batch.memory_entry_opt
        feed_dict[self._memory_size] = batch.memory_size
        feed_dict[self._output_data_type] = batch.output_data_type
        feed_dict[self._output_value] = batch.output_value
        feed_dict[self._output_value_size] = batch.output_value_size
        feed_dict[self._operation] = batch.operation
        feed_dict[self._learning_rate] = batch.learning_rate
        feed_dict[self._dnn_keep_prob] = 1. - self._dropout
        return feed_dict

    def _build_test_feed(self, batch):
        feed_dict = dict()
        feed_dict[self._memory_entry_data_type] = batch.memory_entry_data_type
        feed_dict[self._memory_entry_value] = batch.memory_entry_value
        feed_dict[self._memory_entry_value_size] = batch.memory_entry_value_size
        feed_dict[self._memory_entry_scr_1] = batch.memory_entry_src1
        feed_dict[self._memory_entry_scr_2] = batch.memory_entry_src1
        feed_dict[self._memory_entry_operation] = batch.memory_entry_opt
        feed_dict[self._memory_size] = batch.memory_size
        feed_dict[self._output_data_type] = batch.output_data_type
        feed_dict[self._output_value] = batch.output_value
        feed_dict[self._output_value_size] = batch.output_value_size
        feed_dict[self._dnn_keep_prob] = 1.
        return feed_dict

    def train(self, batch):
        assert not self._is_test
        feed_dict = self._build_train_feed(batch)
        return self._operation_prediction, self._attentive_context_weights, self._loss, self._optimizer, feed_dict

    def predict(self, batch):
        feed_dict = self._build_test_feed(batch)
        return self._operation_prediction, self._attentive_context_weights, feed_dict
