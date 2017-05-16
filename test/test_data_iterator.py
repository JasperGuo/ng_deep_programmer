# coding=utf8

import sys
sys.path += ".."

import os
from learning_models.vocab_manager import VocabManager
from learning_models.basic_model.data_iterator import DataIterator


VOCAB_BASIC_PATH = os.path.join(os.path.abspath(os.curdir), "learning_models", "vocabs")
DATA_PATH = os.path.join(os.path.abspath(os.curdir), "learning_models", "feed_tf")


def main():

    data_type_vocab = VocabManager(os.path.join(VOCAB_BASIC_PATH, "data_type.txt"))
    digit_vocab = VocabManager(os.path.join(VOCAB_BASIC_PATH, "digit.txt"))
    lambda_vocab = VocabManager(os.path.join(VOCAB_BASIC_PATH, "lambda.txt"))
    operation_vocab = VocabManager(os.path.join(VOCAB_BASIC_PATH, "operation.txt"))

    data_iterator = DataIterator(
        data_path=os.path.join(DATA_PATH, "step_1"),
        data_type_vocab=data_type_vocab,
        digit_vocab=digit_vocab,
        lambda_vocab=lambda_vocab,
        operation_vocab=operation_vocab,
        max_memory_size=10,
        max_value_length=20,
        case_num=1,
        batch_size=3
    )
    data_iterator.shuffle()
    data_iterator.get_batch()._print()
    print("Batch per epoch: %d" % data_iterator.batch_per_epoch)
    print("Batch size: %d" % data_iterator.batch_size)
    print("Data Size: %d" % data_iterator.size)


if __name__ == "__main__":
    main()
