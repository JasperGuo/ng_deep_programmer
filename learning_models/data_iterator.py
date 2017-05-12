# coding=utf8

import math
import json
import random
from .vocab_manager import VocabManager


class DataIterator:
    def __init__(self,
                 data_path,
                 data_type_vocab,
                 digit_vocab,
                 lambda_vocab,
                 operation_vocab,
                 batch_size,
                 max_value_length=20,
                 case_num=2
                 ):
        """
        :param data_path:
        :param data_type_vocab:     VocabManager Instance
        :param digit_vocab:         VocabManager Instance
        :param lambda_vocab:        VocabManager Instance
        :param operation_vocab:     VocabManager Instance
        :param batch_size:
        :param case_num:
        """
        self._cursor = 0
        self._digit_vocab = digit_vocab
        self._data_type_vocab = data_type_vocab
        self._operation_vocab = operation_vocab
        self._lambda_vocab = lambda_vocab
        self._case_num = case_num
        self._max_value_length = max_value_length
        self._batch_size = batch_size
        self._data = self._read_data(data_path)
        self._size = len(self._data)
        self._batch_per_epoch = math.floor(self._size / self._batch_size)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def size(self):
        return self._size

    @property
    def batch_per_epoch(self):
        return self._batch_per_epoch

    def shuffle(self):
        random.shuffle(self._data)
        self._cursor = 0

    def _process_memory(self, memory_entry):
        if isinstance(memory_entry["value"], list):
            value = [self._digit_vocab.word2id(v) for v in memory_entry["value"]] + [VocabManager.PAD_TOKEN_ID] * (self._max_value_length - len(memory_entry["value"]))
        else:
            value = [self._digit_vocab.word2id(memory_entry["value"])] + [VocabManager.PAD_TOKEN_ID] * (self._max_value_length - 1)

        for src in ["src1", "src2", "src3"]:
            if memory_entry[src]:
                memory_entry[src] += 1
            else:
                memory_entry[src] = 0

        data_type = self._data_type_vocab.word2id(memory_entry["data_type"])
        opt = self._operation_vocab.word2id(memory_entry["opt"])

        return {
            "value": value,
            "src1": memory_entry["src1"],
            "src2": memory_entry["src2"],
            "src3": memory_entry["src3"],
            "data_type": data_type,
            "opt": opt
        }

    def _process_output(self, output):
        if isinstance(output["value"], list):
            value = [self._digit_vocab.word2id(v) for v in output["value"]] + [VocabManager.PAD_TOKEN_ID] * (self._max_value_length - len(output["value"]))
        else:
            value = [self._digit_vocab.word2id(output["value"])] + [VocabManager.PAD_TOKEN_ID] * (self._max_value_length - 1)

        data_type = self._data_type_vocab.word2id(output["data_type"])

        return {
            "value": value,
            "data_type": data_type
        }

    def _read_data(self, data_path):
        new_data = list()
        with open(data_path, "r") as f:
            data = json.load(f)
        for sample in data:
            detail = random.sample(sample["detail"], self._case_num)
            formatted_detail = []
            for d in detail:
                processed_memory = list()
                for m in d["memory"]:
                    _m = self._process_memory(m)
                    processed_memory.append(_m)
                processed_output = self._process_output(d["output"])
                formatted_detail.append({
                    "memory": processed_memory,
                    "output_value": processed_output["value"],
                    "output_data_type": processed_output["data_type"]
                })
            func = self._operation_vocab.word2id(detail[0]["func"])
            new_data.append({
                "detail": formatted_detail,
                "args": detail[0]["args"],
                "func": func
            })
        return new_data

    def get_batch(self):
        pass
