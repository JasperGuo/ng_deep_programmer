# coding=utf8


class MemoryEntry:
    """
    Memory Entry
    """

    def __init__(self, name, value, data_type, opt, src1=None, src2=None, src3=None, src4=None):
        self._name = name
        self._value = value
        self._type = data_type
        self._src1 = src1
        self._src2 = src2
        self._src3 = src3
        self._src4 = src4
        self._operation = opt

    @property
    def value(self):
        return self._value

    def __str__(self):

        value_str = str(self._value)

        main_content = ', '.join(["name: " + self._name, "value: " + value_str, "type: " + self._type])
        return "MemoryEntry(" + main_content + ")"

    def __repr__(self):
        return self.__str__()


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
