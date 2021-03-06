# coding=utf8

import sys
sys.path.insert(0, "..")

import tensorflow as tf
import util
import models_util


class RNNBasicModel:
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

        self._batch_size = util.get_value(opts, "batch_size")

        self._case_num = util.get_value(opts, "case_num")

        self._batch_with_case_size = self._batch_size * self._case_num

        self._batch_with_case_and_memory_size = self._batch_with_case_size * self._max_memory_size

        self._dropout = util.get_value(opts, "dropout", 0.25)

        # Embedding Size
        self._data_type_embedding_dim = util.get_value(opts, "data_type_embedding_dim")
        self._operation_embedding_dim = util.get_value(opts, "operation_embedding_dim")
        self._digit_embedding_dim = util.get_value(opts, "digit_embedding_dim")

        # Memory Encoder Hidden Unit Size
        self._value_encoder_hidden_dim = util.get_value(opts, "value_encoder_hidden_dim")
        self._value_encoder_rnn_layers = util.get_value(opts, "value_encoder_rnn_layers")

        # Guide Hidden Layer
        self._guide_hidden_dim = util.get_value(opts, "guide_hidden_dim")

        # Operation Selector Hidden Size
        self._operation_selector_dim = util.get_value(opts, "operation_selector_dim")

        self._gradient_clip = util.get_value(opts, "gradient_clip", 5)

        self._build_graph()

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
            self._memory_size = tf.placeholder(tf.int32, [self._batch_with_case_size], name="memory_size")
            self._output_data_type = tf.placeholder(tf.int32, [self._batch_with_case_size], name="output_data_type")
            self._output_value = tf.placeholder(tf.int32, [self._batch_with_case_size, self._max_value_size],
                                                name="output_value")
            self._output_value_size = tf.placeholder(tf.int32, [self._batch_with_case_size], name="output_value_size")

            self._rnn_output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")
            self._rnn_input_keep_prob = tf.placeholder(tf.float32, name="input_keep_prob")

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

        return data_type_embedding, digit_embedding

    def _build_memory_encoder(self):
        with tf.variable_scope("memory_cell"):
            encoder_cell = tf.contrib.rnn.LSTMCell(
                num_units=self._value_encoder_hidden_dim,
                state_is_tuple=True
            )
            encoder_cell = tf.contrib.rnn.DropoutWrapper(
                cell=encoder_cell,
                input_keep_prob=self._rnn_input_keep_prob,
                output_keep_prob=self._rnn_output_keep_prob
            )
            encoder_cell = tf.contrib.rnn.MultiRNNCell(
                [encoder_cell] * self._value_encoder_rnn_layers,
                state_is_tuple=True
            )

        return encoder_cell

    def _build_output_encoder(self):
        with tf.variable_scope("output_cell"):
            encoder_cell = tf.contrib.rnn.LSTMCell(
                num_units=self._value_encoder_hidden_dim,
                state_is_tuple=True
            )
            encoder_cell = tf.contrib.rnn.DropoutWrapper(
                cell=encoder_cell,
                input_keep_prob=self._rnn_input_keep_prob,
                output_keep_prob=self._rnn_output_keep_prob
            )
            encoder_cell = tf.contrib.rnn.MultiRNNCell(
                [encoder_cell] * self._value_encoder_rnn_layers,
                state_is_tuple=True
            )
        return encoder_cell

    def _encode_memory(self, encoder_cell, data_type_embedded, value_embedded, value_size, name):
        """
        Encode memory
        :param encoder_cell:                LSTMCell
        :param data_type_embedded:          [batch_size*case_num*max_memory_size, data_type_embedding_dim]
        :param value_embedded:              [batch_size*case_num*max_memory_size, max_value_size, digit_embedding_dim]
        :param value_size:                  [batch_size*case_num*max_memory_size]
        :param name                         name_scope
        :return:
            [batch_size*case_num*max_memory_size, data_type_embedding_dim+value_encoder_hidden_dim]
        """
        with tf.variable_scope(name):
            # Shape: [batch_size*case_num*max_memory_size, memory_encoder_hidden_dim]
            encoded_value, encoded_states = tf.nn.dynamic_rnn(
                cell=encoder_cell,
                inputs=value_embedded,
                dtype=tf.float32,
                sequence_length=value_size
            )

        indices = tf.add(
            tf.cast(tf.equal(value_size, 0), dtype=tf.int32),
            value_size
        )

        value = tf.reshape(
            models_util.get_last_relevant(
                encoded_value,
                indices
            ),
            shape=[self._batch_with_case_and_memory_size, self._value_encoder_hidden_dim]
        )

        outputs = tf.concat(
            [
                data_type_embedded,
                value
            ],
            axis=1
        )

        return outputs

    def _encode_output(self, encoder_cell, data_type_embedded, value_embedded, value_size, name):
        """
        Encode memory
        :param encoder_cell:                LSTMCell
        :param data_type_embedded:          [batch_size*case_num, data_type_embedding_dim]
        :param value_embedded:              [batch_size*case_num, max_value_size, digit_embedding_dim]
        :param value_size:                  [batch_size*case_num]
        :param name                         name_scope
        :return:
            [batch_size*case_num, data_type_embedding_dim+value_encoder_hidden_dim]
        """
        with tf.variable_scope(name):
            # Shape: [batch_size*case_num*max_memory_size, memory_encoder_hidden_dim]
            encoded_value, encoded_states = tf.nn.dynamic_rnn(
                cell=encoder_cell,
                inputs=value_embedded,
                dtype=tf.float32,
                sequence_length=value_size
            )

        indices = tf.add(
            tf.cast(tf.equal(value_size, 0), dtype=tf.int32),
            value_size
        )

        value = tf.reshape(
            models_util.get_last_relevant(
                encoded_value,
                indices
            ),
            shape=[self._batch_with_case_size, self._value_encoder_hidden_dim]
        )

        outputs = tf.concat(
            [
                data_type_embedded,
                value
            ],
            axis=1
        )

        return outputs

    def _build_memory_output_attention_layer(self, size):
        with tf.variable_scope("memory_output_attention"):
            weights = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[size,
                       size],
                name="weights"
            )

            bias = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[size],
                name="bias"
            )
            return weights, bias

    def _build_attention_score_layer(self):
        with tf.variable_scope("memory_output_attention_score"):
            pass

    def _calc_memory_output_attention(self, weights, bias, encoded_memory, encoded_output, memory_size):
        """
        Calculate Memory, Output Attention
        :param weights:             [data_type_embedding_dim+value_encoder_hidden_dim, data_type_embedding_dim+value_encoder_hidden_dim]
        :param bias:                [data_type_embedding_dim+value_encoder_hidden_dim]
        :param encoded_memory:      [batch_size*case_num*max_memory_size, value_encoder_hidden_dim]
        :param encoded_output:      [batch_size*case_num, value_encoder_hidden_dim]
        :param memory_size:         [batch_size*case_num*max_memory_size]
        :return:
            [batch_size*case_num, data_type_embedding_dim+value_encoder_hidden_dim]
        """

        encoded_dim = self._data_type_embedding_dim + self._value_encoder_hidden_dim

        # Replicate encoded_output
        # Shape: [batch_size*case_num*max_memory_size, output_encoder_layer_2_dim]
        replicated_encoded_output = tf.reshape(
            tf.reshape(
                tf.tile(encoded_output, [1, self._max_memory_size]),
                shape=[self._batch_with_case_size, self._max_memory_size, encoded_dim]
            ),
            shape=[self._batch_with_case_and_memory_size, encoded_dim]
        )

        # Calculate Attention Score
        # Shape: [batch_size*case_num, max_memory_size]
        scores = tf.reshape(
            tf.reduce_sum(
                tf.multiply(
                    encoded_memory,
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
            shape=[self._batch_with_case_size, self._max_memory_size, encoded_dim]
        )

        # Shape: [batch_size*case_num, data_type_embedding_dim+value_encoder_hidden_dim]
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
                    context_vector, weights, transpose_b=True
                ),
                bias
            )
        )
        return attentive_context_vector

    def _build_guide_layer(self):
        encoded_dim = self._data_type_embedding_dim + self._value_encoder_hidden_dim
        with tf.variable_scope("guide"):
            attentive_context_weights = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._guide_hidden_dim,
                       encoded_dim],
                name="attentive_context_weights"
            )
            output_weights = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._guide_hidden_dim,
                       encoded_dim],
                name="output_weights"
            )
            bias = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[self._guide_hidden_dim],
                name="bias"
            )
            return attentive_context_weights, output_weights, bias

    def _calc_guide_vector(self, guide_context_weights, guide_output_weights, guide_bias, context_vector, encoded_output):
        """
        Calculate Guide Vector
        :param guide_context_weights:   [guide_hidden_dim, data_type_embedding+value_encoder_hidden_dim]
        :param guide_output_weights:    [guide_hidden_dim, data_type_embedding+value_encoder_hidden_dim]
        :param guide_bias:              [guide_hidden_dim]
        :param context_vector:          [batch_size*case_num, data_type_embedding+value_encoder_hidden_dim]
        :param encoded_output:          [batch_size*case_num, data_type_embedding+value_encoder_hidden_dim]
        :return:
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

            max_pooling_weights = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self._operation_selector_dim,
                       self._operation_selector_dim],
                name="max_pooling_weights"
            )

            weights = {
                "W1": layer_1_weights,
                "max_pooling_W": max_pooling_weights,
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
                    tf.tanh(
                        tf.matmul(
                            layer_1,
                            selector_weights["max_pooling_W"]
                        ),
                    ),
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

    def _build_graph(self):
        """
        Build Train Computation Graph
        :return:
        """
        self._build_input_nodes()

        data_type_embedding, digit_embedding = self._build_embedding_layer()

        memory_entry_data_type_embedded = tf.nn.embedding_lookup(data_type_embedding, self._memory_entry_data_type)
        memory_entry_value_embedded = tf.nn.embedding_lookup(digit_embedding, self._memory_entry_value)
        output_data_type_embedded = tf.nn.embedding_lookup(data_type_embedding, self._output_data_type)
        output_value_embedded = tf.nn.embedding_lookup(digit_embedding, self._output_value)

        memory_encoder_cell = self._build_memory_encoder()
        output_encoder_cell = self._build_output_encoder()

        # Shape: [batch_size*case_num*max_memory_size, data_type_embedding_dim+value_encoder_hidden_dim]
        encoded_memory = self._encode_memory(
            encoder_cell=memory_encoder_cell,
            data_type_embedded=memory_entry_data_type_embedded,
            value_embedded=memory_entry_value_embedded,
            value_size=self._memory_entry_value_size,
            name="encode_memory"
        )

        # Shape: [batch_size*case_num*max_memory_size, data_type_embedding_dim+value_encoder_hidden_dim]
        encoded_output = self._encode_output(
            encoder_cell=output_encoder_cell,
            data_type_embedded=output_data_type_embedded,
            value_embedded=output_value_embedded,
            value_size=self._output_value_size,
            name="encode_output"
        )

        memory_output_attention_weights, memory_output_attention_bias = self._build_memory_output_attention_layer(
            size=self._value_encoder_hidden_dim+self._data_type_embedding_dim
        )

        attentive_context_vector = self._calc_memory_output_attention(
            weights=memory_output_attention_weights,
            bias=memory_output_attention_bias,
            encoded_memory=encoded_memory,
            encoded_output=encoded_output,
            memory_size=self._memory_size
        )

        guide_context_weights, guide_output_weights, guide_bias = self._build_guide_layer()
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

        self._predicted_operations = selected_operations

        if not self._is_test:
            # Build Loss
            with tf.name_scope("loss"):
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

                self._loss = tf.negative(
                    tf.reduce_mean(log_probs)
                )

            with tf.name_scope("back_propagation"):
                optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
                # self._optimizer = optimizer.minimize(self._loss)

                # clipped at 5 to alleviate the exploding gradient problem
                self._gvs = optimizer.compute_gradients(self._loss)
                self._capped_gvs = [(tf.clip_by_value(grad, -self._gradient_clip, self._gradient_clip), var) for
                                    grad, var
                                    in self._gvs]
                self._optimizer = optimizer.apply_gradients(self._capped_gvs)

    def _build_train_feed(self, batch):
        feed_dict = dict()
        feed_dict[self._memory_entry_data_type] = batch.memory_entry_data_type
        feed_dict[self._memory_entry_value] = batch.memory_entry_value
        feed_dict[self._memory_entry_value_size] = batch.memory_entry_value_size
        feed_dict[self._memory_size] = batch.memory_size
        feed_dict[self._output_data_type] = batch.output_data_type
        feed_dict[self._output_value] = batch.output_value
        feed_dict[self._output_value_size] = batch.output_value_size
        feed_dict[self._rnn_output_keep_prob] = 1. - self._dropout
        feed_dict[self._rnn_input_keep_prob] = 1. - self._dropout
        feed_dict[self._operation] = batch.operation
        feed_dict[self._learning_rate] = batch.learning_rate
        return feed_dict

    def _build_test_feed(self, batch):
        feed_dict = dict()
        feed_dict[self._memory_entry_data_type] = batch.memory_entry_data_type
        feed_dict[self._memory_entry_value] = batch.memory_entry_value
        feed_dict[self._memory_entry_value_size] = batch.memory_entry_value_size
        feed_dict[self._memory_size] = batch.memory_size
        feed_dict[self._output_data_type] = batch.output_data_type
        feed_dict[self._output_value] = batch.output_value
        feed_dict[self._output_value_size] = batch.output_value_size
        feed_dict[self._rnn_output_keep_prob] = 1.
        feed_dict[self._rnn_input_keep_prob] = 1.
        return feed_dict

    def train(self, batch):
        assert not self._is_test
        feed_dict = self._build_train_feed(batch)
        return self._predicted_operations, self._loss, self._optimizer, feed_dict

    def predict(self, batch):
        feed_dict = self._build_test_feed(batch)
        return self._predicted_operations, feed_dict
