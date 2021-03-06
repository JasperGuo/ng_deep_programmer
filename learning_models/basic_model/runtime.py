# coding=utf8

import sys
sys.path.append("..")

import os
import json
import time
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from model import BasicModel
from data_iterator import DataIterator
from vocab_manager import VocabManager


def read_configuration(path):
    with open(path, "r") as f:
        return json.load(f)


class ModelRuntime:

    def __init__(self, configuration):

        self._base_path = os.path.pardir

        self._conf = read_configuration(configuration)

        self._lambda_vocab = VocabManager(
            os.path.join(self._base_path, self._conf["lambda_vocab_file"])
        )
        self._data_type_vocab = VocabManager(
            os.path.join(self._base_path, self._conf["data_type_vocab_file"])
        )
        self._operation_vocab = VocabManager(
            os.path.join(self._base_path, self._conf["operation_vocab_file"])
        )
        self._digit_vocab = VocabManager(
            os.path.join(self._base_path, self._conf["digit_vocab_file"])
        )

        self._epoches = self._conf["epoches"]
        self._batch_size = self._conf["batch_size"]
        self._max_argument_num = self._conf["max_argument_num"]
        self._max_memory_size = self._conf["max_memory_size"]
        self._max_value_size = self._conf["max_value_size"]
        self._case_num = self._conf["case_num"]

        # Learning Rate Strategy
        self._default_learning_rate = self._conf["default_learning_rate"]
        self._learning_rate_decay_interval = self._conf["learning_rate_decay_interval"]
        self._learning_rate_decay_factor = self._conf["learning_rate_decay_factor"]

        self._curr_time = str(int(time.time()))
        self._log_dir = os.path.abspath(self._conf["log_dir"])

        self._result_log_base_path = os.path.abspath(os.path.join(os.path.curdir, self._conf["result_log"], self._curr_time))
        self._checkpoint_path = os.path.abspath(os.path.join(os.path.curdir, self._conf["checkpoint_path"], self._curr_time))
        self._checkpoint_file = os.path.join(os.path.curdir, self._checkpoint_path, "tf_checkpoint")
        self._best_checkpoint_file = os.path.join(os.path.curdir, self._checkpoint_path, "tf_best_checkpoint")

        os.mkdir(self._checkpoint_path)
        self._is_test_capability = self._conf["is_test_capability"]

        os.mkdir(self._result_log_base_path)
        self._save_conf_file(self._result_log_base_path)
        self._save_conf_file(self._checkpoint_path)

        self._test_data_iterator = None
        self._train_data_iterator = None
        self._dev_data_iterator = None

    def _save_conf_file(self, base_path):
        """
        Save Configuration to result log directory
        :return:
        """
        path = os.path.join(base_path, "config.json")
        with open(path, "w") as f:
            f.write(json.dumps(self._conf, indent=4, sort_keys=True))

    def epoch_log(self, file, num_epoch, train_accuracy, dev_accuracy, average_loss):
        """
        Log epoch
        :param file:
        :param num_epoch:
        :param train_accuracy:
        :param dev_accuracy:
        :param average_loss:
        :return:
        """
        with open(file, "a") as f:
            f.write("epoch: %d, train_accuracy: %f, dev_accuracy: %f, average_loss: %f\n" % (num_epoch, train_accuracy, dev_accuracy, average_loss))

    def log(self, file, batch, operation_predictions, argument_predictions):
        pass

    def init_session(self, checkpoint=None):
        self._session = tf.Session()

        with tf.variable_scope("deep_programmer") as scope:
            self._train_model = BasicModel(
                data_type_vocab_manager=self._data_type_vocab,
                operation_vocab_manager=self._operation_vocab,
                digit_vocab_manager=self._digit_vocab,
                lambda_vocab_manager=self._lambda_vocab,
                opts=self._conf,
                is_test=False
            )
            scope.reuse_variables()
            self._test_model = BasicModel(
                data_type_vocab_manager=self._data_type_vocab,
                operation_vocab_manager=self._operation_vocab,
                digit_vocab_manager=self._digit_vocab,
                lambda_vocab_manager=self._lambda_vocab,
                opts=self._conf,
                is_test=True
            )
            self._saver = tf.train.Saver()
            if not checkpoint:
                init = tf.global_variables_initializer()
                self._session.run(init)
            else:
                self._saver.restore(self._session, checkpoint)
            self._file_writer = tf.summary.FileWriter(self._log_dir, self._session.graph)

    def _calc_batch_accuracy(self, operation_predictions, argument_predictions, truth_operations, truth_arguments):
        """
        Check Batch Accuracy
        :param operation_predictions:  Numpy.ndarray    [batch_size]
        :param argument_predictions:   Numpy.ndarray    [batch_size, max_argument_nums]
        :param truth_operations:       Numpy.ndarray    [batch_size]
        :param truth_arguments:        Numpy.ndarray    [batch_size, max]
        :return:
        """
        _truth_operations = np.array(truth_operations)
        _truth_arguments = np.array(truth_arguments)
        correct = 0
        operation_correct = 0
        argument_correct = 0
        for (opt_pred, args_pred, truth_opt, truth_args) in zip(operation_predictions, argument_predictions, _truth_operations, _truth_arguments):

            opt_correct = False
            arg_correct = False

            if np.subtract(opt_pred, truth_opt) == 0:
                operation_correct += 1
                opt_correct = True

            if np.sum(np.abs(np.subtract(args_pred, truth_args))) == 0:
                argument_correct += 1
                arg_correct = True

            if opt_correct and arg_correct:
                correct += 1

        return correct, operation_correct, argument_correct

    def test(self, data_iterator, is_log=False):
        tqdm.write("Testing...")
        total = 0
        correct = 0
        operation_correct = 0
        argument_correct = 0
        for i in tqdm(range(data_iterator.batch_per_epoch)):
            batch = data_iterator.get_batch()
            operations, arguments, feed_dict = self._test_model.predict(batch)
            operations, arguments = self._session.run((
                operations, arguments
            ), feed_dict=feed_dict)

            batch_correct, batch_operation_correct, batch_argument_correct = self._calc_batch_accuracy(
                operation_predictions=operations,
                argument_predictions=arguments,
                truth_operations=batch.operation,
                truth_arguments=batch.argument_targets
            )

            correct += batch_correct
            operation_correct += batch_operation_correct
            argument_correct += batch_argument_correct

            total += batch.batch_size

        accuracy = correct/total
        operation_accuracy = operation_correct/total
        argument_accuracy = argument_correct/total
        return accuracy, operation_accuracy, argument_accuracy

    def train(self):
        try:
            best_accuracy = 0
            last_updated_epoch = 0
            epoch_log_file = os.path.join(self._result_log_base_path, "epoch_result.log")
            curr_learning_rate = self._default_learning_rate
            for epoch in tqdm(range(self._epoches)):
                self._train_data_iterator.shuffle()
                losses = list()
                total = 0
                train_correct = 0
                train_opt_correct = 0
                train_arg_correct = 0
                for i in tqdm(range(self._train_data_iterator.batch_per_epoch)):
                    batch = self._train_data_iterator.get_batch()
                    batch.learning_rate = curr_learning_rate
                    operations, arguments, loss, optimizer, feed_dict = self._train_model.train(batch)
                    operations, arguments, loss, optimizer = self._session.run((
                        operations, arguments, loss, optimizer,
                    ), feed_dict=feed_dict)
                    losses.append(loss)
                    batch_correct, batch_opt_correct, batch_arg_correct = self._calc_batch_accuracy(
                        operations, arguments, batch.operation, batch.argument_targets
                    )
                    train_correct += batch_correct
                    train_opt_correct += batch_opt_correct
                    train_arg_correct += batch_arg_correct
                    total += batch.batch_size
                # tqdm.write(np.array_str(np.array(losses)))
                average_loss = np.average(np.array(losses))

                tqdm.write("epoch: %d, loss: %f" % (epoch, average_loss))
                train_accuracy = train_correct/total
                train_opt_accuracy = train_opt_correct/total
                train_arg_accuracy = train_arg_correct/total
                tqdm.write(", ".join(['Train', "accuracy: %f, opt_accuracy: %f, arg_accuracy: %f" % (train_accuracy, train_opt_accuracy, train_arg_accuracy)]))

                self._dev_data_iterator.shuffle()

                dev_accuracy, dev_operation_accuracy, dev_arg_accuracy = self.test(self._dev_data_iterator, is_log=False)
                tqdm.write(", ".join(['Dev', "accuracy: %f, opt_accuracy: %f, arg_accuracy: %f" % (dev_accuracy, dev_operation_accuracy, dev_arg_accuracy)]))
                tqdm.write("=================================================================")

                if dev_accuracy > best_accuracy:
                    self._saver.save(self._session, self._best_checkpoint_file)
                    best_accuracy = dev_accuracy

                self.epoch_log(
                    epoch_log_file,
                    num_epoch=epoch,
                    train_accuracy=train_accuracy,
                    dev_accuracy=dev_accuracy,
                    average_loss=average_loss
                )

                # Decay Learning rate
                if epoch - last_updated_epoch >= self._learning_rate_decay_interval:
                    curr_learning_rate = curr_learning_rate * self._learning_rate_decay_factor

            else:
                self._saver.save(self._session, self._checkpoint_file)
        except (KeyboardInterrupt, SystemExit):
            # If the user press Ctrl+C...
            # Save the model
            self._saver.save(self._session, self._checkpoint_file)

    def run(self, is_test=False, is_log=False):
        if is_test:

            self._test_data_iterator = DataIterator(
                data_path=os.path.abspath(os.path.join(self._base_path, self._conf["test_file"])),
                digit_vocab=self._digit_vocab,
                data_type_vocab=self._data_type_vocab,
                operation_vocab=self._operation_vocab,
                lambda_vocab=self._lambda_vocab,
                batch_size=self._batch_size,
                max_argument_num=self._max_argument_num,
                max_memory_size=self._max_memory_size,
                max_value_length=self._max_value_size,
                case_num=self._case_num
            )

            test_accuracy, test_opt_accuracy, test_arg_accuracy = self.test(self._test_data_iterator, is_log=is_log)
            tqdm.write("Test, accuracy: %f, opt_accuracy: %f, arg_accuracy: %f" % (test_accuracy, test_opt_accuracy, test_arg_accuracy))
        else:

            self._train_data_iterator = DataIterator(
                data_path=os.path.abspath(os.path.join(self._base_path, self._conf["train_file"])),
                digit_vocab=self._digit_vocab,
                data_type_vocab=self._data_type_vocab,
                operation_vocab=self._operation_vocab,
                lambda_vocab=self._lambda_vocab,
                batch_size=self._batch_size,
                max_argument_num=self._max_argument_num,
                max_memory_size=self._max_memory_size,
                max_value_length=self._max_value_size,
                case_num=self._case_num
            )

            self._dev_data_iterator = DataIterator(
                data_path=os.path.abspath(os.path.join(self._base_path, self._conf["dev_file"])),
                digit_vocab=self._digit_vocab,
                data_type_vocab=self._data_type_vocab,
                operation_vocab=self._operation_vocab,
                lambda_vocab=self._lambda_vocab,
                batch_size=self._batch_size,
                max_argument_num=self._max_argument_num,
                max_memory_size=self._max_memory_size,
                max_value_length=self._max_value_size,
                case_num=self._case_num
            )

            self.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", help="Configuration File")
    parser.add_argument("--checkpoint", help="Is Checkpoint ? Then checkpoint path ?", required=False)
    parser.add_argument("--test", help="Is test ?", dest="is_test", action="store_true")
    parser.add_argument("--no-test", help="Is test ?", dest="is_test", action="store_false")
    parser.set_defaults(is_test=False)
    parser.add_argument("--log", help="Is log ?", dest="is_log", action="store_true")
    parser.add_argument("--no-log", help="Is log ?", dest="is_log", action="store_false")
    parser.set_defaults(is_log=False)
    args = parser.parse_args()

    print(args.conf, args.checkpoint, args.is_test, args.is_log)

    runtime = ModelRuntime(args.conf)
    runtime.init_session(args.checkpoint)
    runtime.run(args.is_test, args.is_log)

