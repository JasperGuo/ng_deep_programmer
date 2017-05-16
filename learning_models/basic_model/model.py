# coding=utf8

import tensorflow as tf

from learning_models import util
from learning_models import models_util


class BasicModel:
    epsilon = 1e-5

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

        # Argument Selector Hidden Size
        self._argument_selector_dim = util.get_value(opts, "argument_selector_dim")
        self._argument_rnn_layers = util.get_value(opts, "argument_rnn_layers")

        # Argument Target Vocab size
        self._argument_candidate_num = self._lambda_vocab_manager.vocab_len + self._max_memory_size + 3

        self._gradient_clip = util.get_value(opts, "gradient_clip", 5)

    def _build_input_nodes(self):
        with tf.name_scope("model_placeholder"):
            self._memory_entry_data_type = tf.placeholder(tf.int32, [self._batch_with_case_and_memory_size],
                                                          name="memory_entry_data_type")
            self._memory_entry_value = tf.placeholder(tf.int32,
                                                      [self._batch_with_case_and_memory_size, self._max_value_size],
                                                      name="memory_entry_value")
            self._memory_size = tf.placeholder(tf.int32, [self._batch_with_case_size], name="memory_size")
            self._output_data_type = tf.placeholder(tf.int32, [self._batch_with_case_size], name="output_data_type")
            self._output_value = tf.placeholder(tf.int32, [self._batch_with_case_size, self._max_value_size],
                                                name="output_value")

            self._rnn_output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")
            self._rnn_input_keep_prob = tf.placeholder(tf.float32, name="input_keep_prob")

            if not self._is_test:
                self._operation = tf.placeholder(tf.int32, [self._batch_size], name="operation")
                # <S>-arg0-arg1-arg2
                self._argument_inputs = tf.placeholder(tf.int32, [self._batch_size, self._max_argument_num], name="argument_inputs")
                # arg0-arg1-arg3-</S>
                self._argument_targets = tf.placeholder(tf.int32, [self._batch_size, self._max_argument_num], name="argument")
                self._learning_rate = tf.placeholder(tf.float32, name="learning_rate")

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

        with tf.variable_scope("auxiliary_argument_embedding"):
            pad_embedding = tf.get_variable(
                initializer=tf.zeros([1, self._lambda_embedding_dim]),
                name="argument_pad_embedding",
                trainable=False
            )
            begin_embedding = tf.get_variable(
                initializer=tf.truncated_normal(
                    [1, self._lambda_embedding_dim]
                ),
                name="argument_begin_embedding",
                trainable=True
            )
            end_embedding = tf.get_variable(
                initializer=tf.truncated_normal(
                    [1, self._lambda_embedding_dim],
                ),
                name="argument_end_embedding",
                trainable=True
            )
            auxiliary_argument_embedding = tf.concat([pad_embedding, end_embedding, begin_embedding], axis=0)

        return data_type_embedding, operation_embedding, digit_embedding, lambda_embedding, auxiliary_argument_embedding

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
                shape=[self._data_type_embedding_dim + self._digit_embedding_dim * self._max_value_size,
                       self._output_encoder_layer_1_dim],
                name="weights_1"
            )
            layer_1_bias = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[self._output_encoder_layer_1_dim],
                name="bias_1"
            )

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

            weights = {
                "W1": layer_1_weights,
                "W2": layer_2_weights
            }

            bias = {
                "b1": layer_1_bias,
                "b2": layer_2_bias
            }

            return weights, bias

    def _encode_memory(self, weights, biases, data_type_embedded, value_embedded):
        """
        Encode memory entry
        :param weights:
        :param biases:
        :param data_type_embedded: [batch_size*case_num*max_memory_size, data_type_embedding_dim]
        :param value_embedded:     [batch_size*case_num*max_memory_size, max_value_size, digit_embedding_dim]
        :return:
            [batch_size*case_num*max_memory_size, memory_encoder_layer_2_dim]
        """
        # Shape: [batch_size*case_num*max_memory_size, max_value_size*digit_embedding+data_type_embedding_dim]
        concatenated_memory_entry_embedded = tf.concat(
            (
                tf.reshape(value_embedded, shape=[self._batch_with_case_and_memory_size,
                                                  self._max_value_size * self._digit_embedding_dim]),
                data_type_embedded
            ),
            axis=1
        )

        with tf.name_scope("encode_memory"):
            layer_1 = tf.add(tf.matmul(concatenated_memory_entry_embedded, weights["W1"]), biases["b1"])
            layer_1 = tf.nn.relu(layer_1)

            layer_2 = tf.add(tf.matmul(layer_1, weights["W2"]), biases["b2"])
            layer_2 = tf.nn.relu(layer_2)

            return layer_2

    def _encode_output(self, weights, biases, data_type_embedded, value_embedded):
        """
        Encode output
        :param weights:
        :param biases:
        :param data_type_embedded: [batch_size*case_num, data_type_embedding_dim]
        :param value_embedded:     [batch_size*case_num, max_value_size, data_type_embedding_dim]
        :return:
            [batch_size*case_num, output_encoder_layer_2_dim]
        """
        # Shape: [batch_size*case_num, max_value_size*data_type_embedding_dim]
        concatenated_output_embedded = tf.concat(
            (
                tf.reshape(value_embedded,
                           shape=[self._batch_with_case_size, self._max_value_size * self._digit_embedding_dim]),
                data_type_embedded
            ),
            axis=1
        )

        with tf.name_scope("encode_output"):
            layer_1 = tf.add(tf.matmul(concatenated_output_embedded, weights["W1"]), biases["b1"])
            layer_1 = tf.nn.relu(layer_1)

            layer_2 = tf.add(tf.matmul(layer_1, weights["W2"]), biases["b2"])
            layer_2 = tf.nn.relu(layer_2)

            return layer_2

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

            score_weights = tf.get_variable(
                initializer=tf.contrib.layer.xavier_initializer(),
                shape=[self._memory_encoder_layer_2_dim,
                       self._output_encoder_layer_2_dim],
                name="score_weights"
            )

            return weights, bias, score_weights

    def _calc_memory_output_attention(self, activate_weights, activate_bias, score_weights, encoded_memory,
                                      encoded_output, memory_size):
        """
        Calculate Memory, Output Attention
        :param activate_weights:    [memory_encoder_layer_2_dim, memory_encoder_layer_2_dim]
        :param activate_bias:       [memory_encoder_layer_2_dim]
        :param score_weights:       [memory_encoder_layer_2_dim, output_encoder_layer_2_dim]
        :param encoded_memory:      [batch_size*case_num*memory_size, memory_encoder_layer_2_dim]
        :param encoded_output:      [batch_size*case_num, output_encoder_layer_2_dim]
        :param memory_size:         [batch_size*case_num]
        :return:
            [batch_size*case_num, guide_hidden_dim]
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
            tf.reduce_sum(
                tf.multiply(
                    tf.matmul(encoded_memory, score_weights),
                    replicated_encoded_output
                ),
                axis=1
            ),
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
        memory_mask = tf.less(
            memory_template, memory_mask
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
                    context_vector, activate_weights, transpose_b=True
                ),
                activate_bias
            )
        )
        return attentive_context_vector

    def _build_guide_layer(self):
        with tf.variable_scope("guide"):
            attentive_context_weights = tf.get_variable(
                initializer=tf.contrib.layer.xavier_initializer(),
                shape=[self._guide_hidden_dim,
                       self._memory_encoder_layer_2_dim],
                name="attentive_context_weights"
            )
            output_weights = tf.get_variable(
                initializer=tf.contrib.layer.xavier_initializer(),
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
        :param guide_context_weights:   [guide_hidden_dim, memory_encoder_layer_2_dim]
        :param guide_output_weights:    [guide_hidden_dim, output_encoder_layer_2_dim]
        :param guide_bias:              [guide_hidden_dim]
        :param context_vector:          [batch_size*case_num, memory_encoder_layer_2_dim]
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
                       self._operation_vocab_manager.vocab_len],
                name="output_weights"
            )
            output_bias = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[self._operation_vocab_manager.vocab_len],
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

    def _select_operation(self, selector_weights, selector_biases, guide_vector):
        """
        :param selector_weights:
        :param selector_biases:
        :param guide_vector: [batch_size*case_num, guide_hidden_dim]
        :return:
            [batch_size, operation_vocab_len], [batch_size]
        """
        with tf.name_scope("encode_output"):
            # Shape: [batch_size*case_num, operation_selector_dim]
            layer_1 = tf.add(tf.matmul(guide_vector, selector_weights["W1"]), selector_biases["b1"])
            layer_1 = tf.nn.relu(layer_1)

            # Max Pooling
            reshaped_layer_1 = tf.transpose(
                tf.reshape(
                    layer_1,
                    shape=[self._batch_size, self._case_num, self._operation_selector_dim]
                ),
                perm=[0, 2, 1]
            )

            # Shape: [batch_size, operation_selector_dim]
            max_pooling_result = tf.reduce_max(
                reshaped_layer_1,
                axis=2
            )

            output_layer = tf.add(tf.matmul(max_pooling_result, selector_weights["output_W"]),
                                  selector_biases["output_b"])

            softmax_output = tf.nn.softmax(output_layer)
            selection = tf.arg_max(softmax_output, dimension=1)

            return softmax_output, selection

    def _build_argument_selector(self):
        with tf.variable_scope("argument_selector"):
            with tf.variable_scope("cell"):
                argument_cell = tf.contrib.rnn.LSTMCell(
                    num_units=self._argument_selector_dim,
                    state_it_tuple=True
                )
                argument_cell = tf.contrib.rnn.DropoutWrapper(
                    cell=argument_cell,
                    input_keep_prob=self._rnn_input_keep_prob,
                    output_keep_prob=self._rnn_output_keep_prob
                )
                argument_cell = tf.contrib.rnn.MultiRNNCell(
                    [argument_cell] * self._argument_rnn_layers,
                    state_is_tuple=True
                )

            with tf.variable_scope("softmax_weights"):
                weights = tf.get_variable(
                    initializer=tf.contrib.layers.xavier_initializer(),
                    shape=[self._argument_selector_dim,
                           self._argument_candidate_num],
                    name="weights"
                )
                bias = tf.get_variable(
                    initializer=tf.zeros_initializer(),
                    shape=[self._argument_candidate_num],
                    name="bias"
                )

            with tf.variable_scope("memory_projection_weights"):
                projection_weights = tf.get_variable(
                    initializer=tf.contrib.layers.xavier_initializer(),
                    shape=[self._memory_encoder_layer_2_dim,
                           self._lambda_embedding_dim],
                    name="weights"
                )
                projection_bias = tf.get_variable(
                    initializer=tf.zeros_initializer(),
                    shape=[self._lambda_embedding_dim],
                    name="bias"
                )

            with tf.variable_scope("max_pooling_weights"):
                pooling_weights = tf.get_variable(
                    initializer=tf.contrib.layers.xavier_initializer(),
                    shape=[self._argument_selector_dim,
                           self._argument_selector_dim],
                    name="weights"
                )

            return argument_cell, weights, bias, projection_weights, projection_bias, pooling_weights

    def _project_memory_entry(self, weights, bias, encoded_memory):
        """
        Linear Mapping to project memory, and Perform Max pooling
        :param weights:         [memory_encoder_layer_2_dim, lambda_embedding_dim]
        :param bias:            [lambda_embedding_dim]
        :param encoded_memory:  [batch_size*case_num*memory_max_size, memory_encoder_layer_2_dim]
        :return:
            [batch_size, max_memory_size, lambda_embedding_dim]
        """
        projected_memory = tf.add(
            tf.matmul(
                encoded_memory,
                weights
            ),
            bias
        )

        # Max Pooling
        # Shape: [batch_size, max_memory_size, lambda_embedding_dim, case_num]
        reshaped_memory = tf.transpose(
            tf.reshape(
                projected_memory,
                shape=[self._batch_size, self._case_num, self._max_memory_size, self._lambda_embedding_dim]
            ),
            perm=[0, 2, 3, 1]
        )

        # Shape: [batch_size, max_memory_size, lambda_embedding_dim]
        max_pooling_result = tf.reduce_max(
            reshaped_memory,
            axis=3
        )

        return max_pooling_result

    def _build_argument_embedding(self, lambda_embedding, projected_memory, auxiliary_embedding):
        """
        Construct Argument Embedding, append a placeholder memory at last position (indicating NOP)
        :param lambda_embedding:    [lambda_vocab_size, lambda_embedding]
        :param projected_memory:    [batch_size, max_memory_size, projected_memory]
        :param auxiliary_embedding: [2, lambda_embedding]
                                        <BEGIN>
                                        <NOP>
        :return:
            [batch_size, (argument_candidate_num), lambda_embedding]
        """

        # Shape: [lambda_vocab_size + 2, lambda_embedding]
        _lambda_embedding = tf.concat(lambda_embedding, auxiliary_embedding)

        expanded_embedding = tf.tile(
            tf.expand_dims(
                _lambda_embedding,
                dim=0
            ),
            [self._batch_size, 1, 1]
        )
        argument_embedding = tf.concat([projected_memory, expanded_embedding], axis=2)
        return argument_embedding

    def _look_up_argument_embedding(self, argument_embedding, arguments):
        """
        Look up argument_embedding
        :param argument_embedding:  [batch_size, (argument_candidate_num), lambda_embedding_dim]
        :param arguments:           [batch_size, None]
        :return:
            [batch_size, None, lambda_embedding_dim]
        """
        # [batch_size]
        indices_template = tf.range(self._batch_size) * self._argument_candidate_num
        # [batch_size, max_argument_num]
        indices = tf.add(
            arguments,
            tf.reshape(
                indices_template,
                shape=[self._batch_size, 1]
            )
        )
        argument_embedded = tf.nn.embedding_lookup(
            tf.reshape(
                argument_embedding,
                shape=[self._batch_size * self._argument_candidate_num, self._lambda_embedding_dim]
            ),
            indices
        )
        return argument_embedded

    def _select_arguments_in_training(
            self,
            argument_cell,
            guide_vector,
            selected_operation,
            argument_embedding,
            arguments,
            max_pooling_weights,
            softmax_weights,
            softmax_bias
    ):
        """
        Select arguments in training process
        :param argument_cell:       LSTMCell
        :param guide_vector:        [batch_size*case_num, guide_hidden_dim]
        :param selected_operation:  [batch_size, operation_selector_dim]
        :param argument_embedding:  [batch_size, (argument_candidate_num), lambda_embedding_dim]
        :param arguments:           [batch_size, max_argument_num]:
        :param max_pooling_weights: [argument_selector_hidden_dim, argument_selector_hidden_dim],
        :param softmax_weights:           []
        :param softmax_bias:
        :return:
            [batch_size, max_argument_num]
        """
        assert not self._is_test
        # [batch_size, max_argument_num, lambda_embedding]
        argument_embedded = self._look_up_argument_embedding(
            argument_embedding=argument_embedding,
            arguments=arguments
        )

        # Replicate argument embedded
        replicated_argument_embedded = tf.reshape(
            tf.tile(
                argument_embedded,
                [1, self._case_num, 1]
            ),
            shape=[self._batch_with_case_size, self._max_argument_num, self._lambda_embedding_dim]
        )

        # Replicate operation embedding
        replicated_operation_embedded = tf.reshape(
            tf.tile(
                selected_operation,
                [1, self._case_num * self._max_argument_num]
            ),
            shape=[self._batch_with_case_size, self._max_argument_num, self._operation_embedding_dim]
        )

        # Replicate guide embedding
        replicated_guide_vector = tf.reshape(
            tf.tile(
                guide_vector,
                [1, self._max_argument_num]
            ),
            shape=[self._batch_with_case_size, self._max_argument_num, self._guide_hidden_dim]
        )

        # Shape: [batch_size*case_num, max_argument_num, guide_hidden_dim+operation_selector_dim+lambda_embedding_dim]
        inputs = tf.concat(
            [
                replicated_guide_vector,
                replicated_operation_embedded,
                replicated_argument_embedded
            ],
            axis=2
        )

        # Argument_outputs Shape: [batch_size*case_num, max_argument_num, argument_selector_dim]
        argument_outputs, argument_states = tf.nn.dynamic_rnn(
            cell=argument_cell,
            inputs=inputs,
            dtype=tf.float32
        )

        # Max Pooling
        # Shape: [batch_size*case_num*max_argument_num, argument_selector_dim]
        weighted_argument_outputs = tf.tanh(
            tf.matmul(
                tf.reshape(
                    argument_outputs,
                    shape=[self._batch_with_case_size * self._max_argument_num, self._argument_selector_dim]
                ),
                max_pooling_weights,
            )
        )

        # Shape: [batch_size, max_argument_num, argument_selector_dim]
        max_pooling_results = tf.reduce_max(
            # Shape: [batch_size, max_argument_num, argument_selector_dim, case_num]
            tf.transpose(
                tf.reshape(
                    weighted_argument_outputs,
                    shape=[self._batch_size, self._case_num, self._max_argument_num, self._argument_selector_dim]
                ),
                perm=[0, 2, 3, 1]
            ),
            axis=3
        )

        # Shape: [batch_size, max_argument_num, argument_candidate_num]
        softmax_outputs = tf.nn.softmax(
            tf.reshape(
                tf.add(
                    tf.matmul(
                        tf.reshape(
                            max_pooling_results,
                            shape=[self._batch_size * self._max_argument_num, self._argument_selector_dim]
                        ),
                        softmax_weights,
                    ),
                    softmax_bias
                ),
                shape=[self._batch_size, self._max_argument_num, self._argument_candidate_num]
            ),
            dim=2
        )

        predictions = tf.arg_max(softmax_outputs, dimension=2)

        return softmax_outputs, predictions

    def _select_arguments_in_testing(
            self,
            argument_cell,
            guide_vector,
            selected_operation,
            argument_embedding,
            max_pooling_weights,
            softmax_weights,
            softmax_bias
    ):
        """
        Select arguments in testing process
        :param argument_cell:              LSTMCell
        :param guide_vector:               [batch_size*case_num, guide_hidden_dim]
        :param selected_operation:         [batch_size, operation_selector_dim]
        :param argument_embedding:         [batch_size, (argument_candidate_num), lambda_embedding_dim]
        :param max_pooling_weights:        [argument_selector_hidden_dim, argument_selector_hidden_dim]
        :param softmax_weights:            [argument_selector_dim, argument_candidate_num]
        :param softmax_bias:               [argument_candidate_num]
        :return:
        """
        assert self._is_test

        # Replicate operation embedding
        # Shape: [batch_size*case_num, 1, operation_embedding_dim]
        replicated_operation_embedded = tf.reshape(
            tf.tile(
                selected_operation,
                [1, self._case_num]
            ),
            shape=[self._batch_with_case_size, 1, self._operation_embedding_dim]
        )

        # Shape: [batch_size*case_num, 1, guide_hidden_dim]
        reshaped_guide_vector = tf.reshape(
            guide_vector,
            shape=[self._batch_with_case_size, 1, self._guide_hidden_dim]
        )

        # Shape: [batch_size, 1]
        first_input_index = tf.reshape(
            tf.constant([self._argument_candidate_num - 1] * self._batch_size),
            shape=[self._batch_size, 1]
        )

        # Initialize an zeros states
        initial_state = tf.zeros([self._argument_rnn_layers, 2, self._batch_size, self._argument_selector_dim])
        l = tf.unstack(initial_state, axis=0)
        rnn_state_tuple = tuple(
            [tf.contrib.rnn.LSTMStateTuple(l[idx][0], l[idx][1])
             for idx in range(self._argument_rnn_layers)]
        )

        with tf.name_scope("select_arguments"):
            def __cond(_curr_ts, _inputs, rnn_states, prediction_array):
                return tf.less(_curr_ts, self._max_argument_num)

            def __loop_body(_curr_ts, _inputs, rnn_states, prediction_array):
                """
                :param _curr_ts:            Scalar
                :param _inputs:             [batch_size, 1]
                :param rnn_states:          LSTMStateTuple
                :param prediction_array:    TensorArray
                :return:
                """
                # [batch_size, 1, lambda_embedding_dim]
                _input_embedding = self._look_up_argument_embedding(
                    argument_embedding=argument_embedding,
                    arguments=_inputs
                )

                # Replicate input embedding
                # Shape: [batch_size*case_num, 1, lambda_embedding_dim]
                replicated_input_embedding = tf.reshape(
                    tf.tile(
                        _input_embedding,
                        [1, self._case_num, 1]
                    ),
                    shape=[self._batch_with_case_size, 1, self._lambda_embedding_dim]
                )

                # Shape: [batch_size*case_num, 1, guide_hidden_dim+operation_selector_dim+lambda_embedding_dim]
                concatenated_input_embedding = tf.concat(
                    [
                        reshaped_guide_vector,
                        replicated_operation_embedded,
                        replicated_input_embedding
                    ],
                    axis=2
                )
                # Shape: [batch_size*case_num, 1, argument_selector_dim]
                _argument_rnn_outputs, _argument_rnn_states = tf.nn.dynamic_rnn(
                    cell=argument_cell,
                    inputs=concatenated_input_embedding,
                    dtype=tf.float32
                )

                # Shape: [batch_size*case_num, argument_selector_dim]
                _weighted_argument_outputs = tf.tanh(
                    tf.matmul(
                        tf.reshape(
                            _argument_rnn_outputs,
                            shape=[self._batch_with_case_size * 1, self._argument_selector_dim]
                        ),
                        max_pooling_weights,
                    )
                )

                # Shape: [batch_size, argument_selector_dim]
                _max_pooling_results = tf.reduce_max(
                    # Shape: [batch_size, argument_selector_dim, case_num]
                    tf.transpose(
                        tf.reshape(
                            _weighted_argument_outputs,
                            shape=[self._batch_size, self._case_num, self._argument_selector_dim]
                        ),
                        perm=[0, 2, 1]
                    ),
                    axis=2
                )

                # Shape: [batch_size, argument_candidate_num]
                softmax_outputs = tf.nn.softmax(
                    tf.reshape(
                        tf.add(
                            tf.matmul(
                                tf.reshape(
                                    _max_pooling_results,
                                    shape=[self._batch_size, self._argument_selector_dim]
                                ),
                                softmax_weights,
                            ),
                            softmax_bias
                        ),
                        shape=[self._batch_size, self._argument_candidate_num]
                    ),
                    dim=1
                )

                # Shape: [batch_size, 1]
                curr_prediction = tf.reshape(
                    tf.cast(tf.arg_max(softmax_outputs, dimension=1), dtype=tf.int32),
                    shape=[self._batch_size, 1]
                )
                prediction_array = prediction_array.write(
                    _curr_ts,
                    tf.reshape(
                        curr_prediction,
                        shape=[self._batch_size]
                    ),
                )

                next_ts = tf.add(_curr_ts, 1)

                return next_ts, curr_prediction, _argument_rnn_states, prediction_array

            total_ts, last_prediction, rnn_states, predictions, = tf.while_loop(
                body=__loop_body,
                cond=__cond,
                loop_vars=[
                    tf.constant(0),
                    first_input_index,
                    rnn_state_tuple,
                    tf.TensorArray(dtype=tf.int32, size=self._max_argument_num)
                ]
            )

            return tf.transpose(
                predictions.stack(name="argument_predictions")
            )

    def _build_train_graph(self):
        """
        Build Training Graph
        :return:
        """
        self._build_input_nodes()

        data_type_embedding, operation_embedding, digit_embedding, lambda_embedding, auxiliary_argument_embedding = self._build_embedding_layer()

        memory_entry_data_type_embedded = tf.nn.embedding_lookup(data_type_embedding, self._memory_entry_data_type)
        memory_entry_value_embedded = tf.nn.embedding_lookup(digit_embedding, self._memory_entry_value)
        output_data_type_embedded = tf.nn.embedding_lookup(data_type_embedding, self._output_data_type)
        output_value_embedded = tf.nn.embedding_lookup(digit_embedding, self._output_value)

        memory_encoder_weights, memory_encoder_biases = self._build_memory_encoder()
        output_encoder_weights, output_encoder_biases = self._build_output_encoder()

        # Encode Memory and Output
        encoded_memory = self._encode_memory(memory_encoder_weights, memory_encoder_biases,
                                             memory_entry_data_type_embedded, memory_entry_value_embedded)
        encoded_output = self._encode_output(output_encoder_weights, output_encoder_biases, output_data_type_embedded,
                                             output_value_embedded)

        # Guide
        attention_weights, attention_bias, attention_score_weights = self._build_memory_output_attention_layer()
        attentive_context_vector = self._calc_memory_output_attention(
            activate_weights=attention_weights,
            activate_bias=attention_bias,
            score_weights=attention_score_weights,
            encoded_memory=encoded_memory,
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
            guide_vector=guide_vector
        )

        # Prevent inf
        operation_softmax_output = tf.add(operation_softmax_output, tf.constant(self.epsilon, dtype=tf.float32))

        # Truth embedded operation
        embedd_operation = tf.reshape(
            tf.nn.embedding_lookup(operation_embedding, self._operation),
            shape=[self._batch_size, self._operation_embedding_dim]
        )

        # Argument Selector
        argument_rnn_cell, argument_softmax_weights, argument_softmax_bias, memory_projection_weights, memory_projection_bias, max_pooling_weights = self._build_argument_selector()

        # Project Memory Entry
        # Shape: [batch_size, max_memory_size, lambda_embedding_dim]
        projected_memory = self._project_memory_entry(
            weights=memory_projection_weights,
            bias=memory_projection_bias,
            encoded_memory=encoded_memory
        )

        # Argument Vocab Embedding
        # Shape: [batch_size, argument_candidate_num, lambda_embedding]
        argument_embedding = self._build_argument_embedding(
            lambda_embedding=lambda_embedding,
            projected_memory=projected_memory,
            auxiliary_embedding=auxiliary_argument_embedding
        )

        # argument_outputs Shape:     [batch_size, max_argument_num, argument_candidate_num]
        # argument_predictions Shape: [batch_size, amx_argument_num]
        argument_outputs, argument_predictions = self._select_arguments_in_training(
            argument_cell=argument_rnn_cell,
            guide_vector=guide_vector,
            selected_operation=embedd_operation,
            argument_embedding=argument_embedding,
            arguments=self._argument_inputs,
            max_pooling_weights=max_pooling_weights,
            softmax_weights=argument_softmax_weights,
            softmax_bias=argument_softmax_bias
        )

        # Prevent inf
        argument_outputs = tf.add(argument_outputs, tf.constant(self.epsilon, dtype=tf.float32))

        self._operation_prediction = selected_operations
        self._argument_prediction = argument_predictions

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

            #############################################################################
            # Argument Probs
            # Calculate truth argument Index
            # Shape: [batch_size, max_argument_num]
            first_prefix_index = tf.tile(
                tf.reshape(
                    tf.range(self._batch_size),
                    shape=[self._batch_size, 1]
                ),
                [1, self._max_argument_num]
            )
            second_prefix_index = tf.transpose(
                tf.tile(
                    tf.reshape(
                        tf.range(self._max_argument_num),
                        shape=[self._max_argument_num, 1]
                    ),
                    [1, self._batch_size]
                )
            )
            stacked_prefix = tf.stack([first_prefix_index, second_prefix_index], axis=2)
            truth_argument_indices = tf.concat(
                [
                    stacked_prefix,
                    tf.reshape(self._argument_targets, shape=[self._batch_size, self._max_argument_num, 1])
                ],
                axis=2
            )
            # Shape: [batch_size, max_argument_num]
            truth_argument_probs = tf.gather_nd(argument_outputs, truth_argument_indices)

            #############################################################################

            # [batch_size]
            arguments_log_probs = tf.reduce_sum(
                tf.log(truth_argument_probs), axis=1
            )

            # [batch_size]
            log_probs = tf.add(
                arguments_log_probs, tf.log(truth_operations_probs)
            )

            self._loss = tf.negative(
                tf.reduce_mean(log_probs)
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

        data_type_embedding, operation_embedding, digit_embedding, lambda_embedding, auxiliary_argument_embedding = self._build_embedding_layer()

        memory_entry_data_type_embedded = tf.nn.embedding_lookup(data_type_embedding, self._memory_entry_data_type)
        memory_entry_value_embedded = tf.nn.embedding_lookup(digit_embedding, self._memory_entry_value)
        output_data_type_embedded = tf.nn.embedding_lookup(data_type_embedding, self._output_data_type)
        output_value_embedded = tf.nn.embedding_lookup(digit_embedding, self._output_value)

        memory_encoder_weights, memory_encoder_biases = self._build_memory_encoder()
        output_encoder_weights, output_encoder_biases = self._build_output_encoder()

        # Encode Memory and Output
        encoded_memory = self._encode_memory(memory_encoder_weights, memory_encoder_biases,
                                             memory_entry_data_type_embedded, memory_entry_value_embedded)
        encoded_output = self._encode_output(output_encoder_weights, output_encoder_biases, output_data_type_embedded,
                                             output_value_embedded)

        # Guide
        attention_weights, attention_bias, attention_score_weights = self._build_memory_output_attention_layer()
        attentive_context_vector = self._calc_memory_output_attention(
            activate_weights=attention_weights,
            activate_bias=attention_bias,
            score_weights=attention_score_weights,
            encoded_memory=encoded_memory,
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
            guide_vector=guide_vector
        )

        # Truth embedded operation
        embedd_operation = tf.reshape(
            tf.nn.embedding_lookup(operation_embedding, selected_operations),
            shape=[self._batch_size, self._operation_embedding_dim]
        )

        # Argument Selector
        argument_rnn_cell, argument_softmax_weights, argument_softmax_bias, memory_projection_weights, memory_projection_bias, max_pooling_weights = self._build_argument_selector()

        # Project Memory Entry
        # Shape: [batch_size, max_memory_size, lambda_embedding_dim]
        projected_memory = self._project_memory_entry(
            weights=memory_projection_weights,
            bias=memory_projection_bias,
            encoded_memory=encoded_memory
        )

        # Argument Vocab Embedding
        # Shape: [batch_size, argument_candidate_num, lambda_embedding]
        argument_embedding = self._build_argument_embedding(
            lambda_embedding=lambda_embedding,
            projected_memory=projected_memory,
            auxiliary_embedding=auxiliary_argument_embedding
        )

        # [batch_size, max_argument_num]
        argument_predictions = self._select_arguments_in_testing(
            argument_cell=argument_rnn_cell,
            guide_vector=guide_vector,
            selected_operation=embedd_operation,
            argument_embedding=argument_embedding,
            max_pooling_weights=max_pooling_weights,
            softmax_weights=argument_softmax_weights,
            softmax_bias=argument_softmax_bias
        )

        self._operation_prediction = selected_operations
        self._argument_prediction = argument_predictions

    def _build_train_feed(self, batch):
        feed_dict = dict()
        feed_dict[self._memory_entry_data_type] = batch.memory_entry_data_type
        feed_dict[self._memory_entry_value] = batch.memory_entry_value
        feed_dict[self._memory_size] = batch.memory_size
        feed_dict[self._output_data_type] = batch.output_data_type
        feed_dict[self._output_value] = batch.output_value
        feed_dict[self._rnn_output_keep_prob] = 1. - self._dropout
        feed_dict[self._rnn_input_keep_prob] = 1. - self._dropout
        feed_dict[self._operation] = batch.operation
        feed_dict[self._argument_inputs] = batch.argument_inputs
        feed_dict[self._argument_targets] = batch.argument_targets
        feed_dict[self._learning_rate] = batch.learning_rate
        return feed_dict

    def _build_test_feed(self, batch):
        feed_dict = dict()
        feed_dict[self._memory_entry_data_type] = batch.memory_entry_data_type
        feed_dict[self._memory_entry_value] = batch.memory_entry_value
        feed_dict[self._memory_size] = batch.memory_size
        feed_dict[self._output_data_type] = batch.output_data_type
        feed_dict[self._output_value] = batch.output_value
        feed_dict[self._rnn_output_keep_prob] = 1.
        feed_dict[self._rnn_input_keep_prob] = 1.
        return feed_dict

    def train(self, batch):
        assert not self._is_test
        feed_dict = self._build_train_feed(batch)
        return self._operation_prediction, self._argument_prediction, self._loss, self._optimizer, feed_dict

    def predict(self, batch):
        feed_dict = self._build_test_feed(batch)
        return self._operation_prediction, self._argument_prediction, feed_dict

