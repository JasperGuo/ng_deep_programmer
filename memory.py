# coding=utf8

import json


class MemoryEntry:
    """
    Memory Entry
    """

    def __init__(self, name, value, data_type, opt, lambda_expr=None, src1=None, src2=None):
        self._name = name
        self._value = value
        self._type = data_type
        self._src1 = src1
        self._src2 = src2
        self._operation = opt
        self._lambda_expr = lambda_expr

    @property
    def value(self):
        return self._value

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    def __str__(self):

        value_str = str(self._value)

        main_content = ', '.join(["name: " + self._name, "value: " + value_str, "type: " + self._type])
        return "MemoryEntry(" + main_content + ")"

    def __repr__(self):
        return self.__str__()

    def serialize(self):
        return {
            "variable_name": self._name,
            "value": self._value,
            "data_type": self._type,
            "src1": self._src1.name if self._src1 else None,
            "src2": self._src2.name if self._src2 else None,
            "opt": self._operation,
            "lambda_expr": self._lambda_expr
        }


class Memory:
    def __init__(self):
        self._memory_dict = dict()
        self._memory_list = list()

    def read(self, variable_name):
        if variable_name not in self._memory_dict:
            return None
        return self._memory_dict[variable_name]

    def write(self, variable_name, value):
        """
        :param variable_name:
        :param value: MemoryEntry instance
        :return:
        """
        self._memory_dict[variable_name] = value
        self._memory_list.append(value)

    def __str__(self):
        memory_str = ', '.join([str(me) for me in self._memory_list])
        return memory_str

    def __repr__(self):
        return self.__str__()

    def serialize(self):
        result = list()
        for memory_entry in self._memory_list:
            result.append(memory_entry.serialize())
        return result
