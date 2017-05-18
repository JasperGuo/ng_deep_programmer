# coding=utf8

import sys
sys.path.insert(0, "..")

import math
import json
import random
from vocab_manager import VocabManager


class Batch:
    """
    Batch Data
    """
    def __init__(
            self,
            memory_entry_data_type,
            memory_entry_value,
            memory_entry_value_size,
            memory_entry_opt,
            memory_entry_src1,
            memory_entry_src2,
            memory_entry_lambda,
            memory_size,
            output_data_type,
            output_value,
            output_value_size,
            operation,
            args
    ):
        """
        :param memory_entry_data_type: [batch_size*case_num*max_memory_size]
        :param memory_entry_value:     [batch_size*case_num*max_memory_size, max_value_size]
        :param memory_entry_value_size:[batch_size*case*max_memory_size]
        :param memory_entry_opt:       [batch_size*case_num*max_memory_size]
        :param memory_entry_src1:      [batch_size*case_num*max_memory_size]
        :param memory_entry_src2:      [batch_size*case_num*max_memory_size]
        :param memory_entry_lambda:    [batch_size*case_num*max_memory_size]
        :param memory_size:            [batch_size*case_num]
        :param output_data_type:       [batch_size*case_num]
        :param output_value:           [batch_size*case_num]
        :param operation               [batch_size]
        :param args:                   [batch_size, max_argument_num]
        """
        self.memory_entry_data_type = memory_entry_data_type
        self.memory_entry_value = memory_entry_value
        self.memory_entry_value_size = memory_entry_value_size
        self.memory_entry_opt = memory_entry_opt
        self.memory_entry_src1 = memory_entry_src1
        self.memory_entry_src2 = memory_entry_src2
        self.memory_entry_lambda = memory_entry_lambda
        self.memory_size = memory_size
        self.output_data_type = output_data_type
        self.output_value = output_value
        self.output_value_size = output_value_size
        self.operation = operation

        self.argument_inputs = list()
        self.argument_targets = list()
        for arg in args:
            self.argument_inputs.append(arg[:len(arg)-1])
            self.argument_targets.append(arg[1::])

        self._learning_rate = 0.0

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, rate):
        self._learning_rate = rate

    @property
    def batch_size(self):
        return len(self.output_value)

    def _print(self):
        print("Output Value: ")
        print(self.output_value)
        print("Output Data Type: ")
        print(self.output_data_type)
        print("Memory Size: ")
        print(self.memory_size)
        print("Memory Entry Value: ")
        print(self.memory_entry_value)
        print("Memory Data Type: ")
        print(self.memory_entry_data_type)
        print("Memory Entry Operation: ")
        print(self.memory_entry_opt)
        print("Memory Entry Src1: ")
        print(self.memory_entry_src1)
        print("Memory Entry Src2: ")
        print(self.memory_entry_src2)
        print("Memory Entry Lambda: ")
        print(self.memory_entry_lambda)
        print("Operation: ")
        print(self.operation)
        print("Argument inputs: ")
        print(self.argument_inputs)
        print("Argument targets: ")
        print(self.argument_targets)
        print("Learning rate: ")
        print(self.learning_rate)


class DataIterator:
    def __init__(self,
                 data_path,
                 data_type_vocab,
                 digit_vocab,
                 lambda_vocab,
                 operation_vocab,
                 batch_size,
                 max_argument_num=4,
                 max_memory_size=10,
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
        :param max_argument_num
        :param max_memory_size:
        :param max_value_length:
        :param case_num:
        """
        self._cursor = 0
        self._digit_vocab = digit_vocab
        self._data_type_vocab = data_type_vocab
        self._operation_vocab = operation_vocab
        self._lambda_vocab = lambda_vocab
        self._case_num = case_num
        self._max_argument_num = max_argument_num
        self._max_value_length = max_value_length
        self._max_memory_size = max_memory_size
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
            value_size = len(memory_entry["value"])
            value = [self._digit_vocab.word2id(v) for v in memory_entry["value"]] + [VocabManager.PAD_TOKEN_ID] * (self._max_value_length - len(memory_entry["value"]))
        else:
            value_size = 1
            value = [self._digit_vocab.word2id(memory_entry["value"])] + [VocabManager.PAD_TOKEN_ID] * (self._max_value_length - 1)

        for src in ["src1", "src2"]:
            if memory_entry[src]:
                memory_entry[src] += 1
            else:
                memory_entry[src] = 0

        data_type = self._data_type_vocab.word2id(memory_entry["data_type"])
        opt = self._operation_vocab.word2id(memory_entry["opt"]) if memory_entry["opt"] else self._operation_vocab.word2id("NOP")
        lambda_expr = self._lambda_vocab.word2id(memory_entry["lambda_expr"]) if memory_entry["lambda_expr"] else self._lambda_vocab.word2id("NOP")

        return {
            "value": value,
            "value_size": value_size,
            "src1": memory_entry["src1"],
            "src2": memory_entry["src2"],
            "data_type": data_type,
            "opt": opt,
            "lambda_expr": lambda_expr
        }

    def _pad_memory(self, memory):
        """
        Pad memory to meet max memory size
        :param memory:
        :return:
        """
        curr_length = len(memory)
        diff = self._max_memory_size - curr_length
        for i in range(diff):
            memory.append({
                "value": [VocabManager.PAD_TOKEN_ID] * self._max_value_length,
                "value_size": 0,
                "src1": 0,
                "src2": 0,
                "data_type": VocabManager.PAD_TOKEN_ID,
                "opt": VocabManager.PAD_TOKEN_ID,
                "lambda_expr": VocabManager.PAD_TOKEN_ID
            })

    def _process_output(self, output):
        if isinstance(output["value"], list):
            value_size = len(output["value"])
            value = [self._digit_vocab.word2id(v) for v in output["value"]] + [VocabManager.PAD_TOKEN_ID] * (self._max_value_length - len(output["value"]))
        else:
            value_size = 1
            value = [self._digit_vocab.word2id(output["value"])] + [VocabManager.PAD_TOKEN_ID] * (self._max_value_length - 1)

        data_type = self._data_type_vocab.word2id(output["data_type"])

        return {
            "value_size": value_size,
            "value": value,
            "data_type": data_type
        }

    def _process_args(self, args):

        # Argument Begin Word id
        BEGIN_ID = self._max_memory_size+self._lambda_vocab.vocab_len + 2

        # Argument END Word id
        END_ID = self._max_memory_size+self._lambda_vocab.vocab_len + 1

        # Argument PAD Word id
        PAD_ID = self._max_memory_size+self._lambda_vocab.vocab_len

        processed_args = [BEGIN_ID]
        for arg in args:
            if not arg:
                break
            else:
                if arg[1] == "FUNCTION":
                    # lambda
                    arg_id = self._max_memory_size + self._lambda_vocab.word2id(arg[0])
                else:
                    arg_id = arg[0]
                processed_args.append(arg_id)

        i = len(processed_args)
        while i < self._max_argument_num:
            processed_args.append(PAD_ID)
            i += 1
        processed_args.append(END_ID)
        return processed_args

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
                actual_memory_size = len(processed_memory)
                self._pad_memory(processed_memory)
                processed_output = self._process_output(d["output"])
                formatted_detail.append({
                    "memory_size": actual_memory_size,
                    "memory": processed_memory,
                    "output_value": processed_output["value"],
                    "output_value_size": processed_output["value_size"],
                    "output_data_type": processed_output["data_type"]
                })
            func = self._operation_vocab.word2id(detail[0]["func"])

            args = self._process_args(detail[0]["args"])

            new_data.append({
                "detail": formatted_detail,
                "args": args,
                "func": func
            })
        return new_data

    def get_batch(self):

        if self._cursor + self._batch_size > self._size:
            raise IndexError("Index Error")

        samples = self._data[self._cursor:self._cursor+self._batch_size]
        self._cursor += self._batch_size

        # [batch_size * case_num * max_memory_length]
        memory_entry_data_type = list()
        memory_entry_value = list()
        memory_entry_value_size = list()
        memory_entry_opt = list()
        memory_entry_src1 = list()
        memory_entry_src2 = list()
        memory_entry_lambda = list()

        # [batch_size * case_num]
        memory_size = list()

        # [batch_size * case_num]
        output_data_type = list()
        output_value = list()
        output_value_size = list()

        func = list()
        args = list()

        for s in samples:
            func.append(s["func"])
            args.append(s["args"])
            for case in s["detail"]:
                memory_size.append(case["memory_size"])
                output_data_type.append(case["output_data_type"])
                output_value.append(case["output_value"])
                output_value_size.append(case["output_value_size"])
                for memory_entry in case["memory"]:
                    memory_entry_data_type.append(memory_entry["data_type"])
                    memory_entry_value.append(memory_entry["value"])
                    memory_entry_value_size.append(memory_entry["value_size"])
                    memory_entry_opt.append(memory_entry["opt"])
                    memory_entry_src1.append(memory_entry["src1"])
                    memory_entry_src2.append(memory_entry["src2"])
                    memory_entry_lambda.append(memory_entry["lambda_expr"])

        return Batch(
            memory_entry_data_type=memory_entry_data_type,
            memory_entry_value=memory_entry_value,
            memory_entry_value_size=memory_entry_value_size,
            memory_entry_src1=memory_entry_src1,
            memory_entry_src2=memory_entry_src2,
            memory_entry_opt=memory_entry_opt,
            memory_entry_lambda=memory_entry_lambda,
            memory_size=memory_size,
            output_data_type=output_data_type,
            output_value=output_value,
            output_value_size=output_value_size,
            operation=func,
            args=args
        )
