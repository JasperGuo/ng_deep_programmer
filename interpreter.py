# coding=utf8

from program_generator.func_set import FUNCTIONS, FunctionType, DataType
from memory import MemoryEntry, Memory
import program_settings as ps


def interpret(memory, expression):
    """
    interpret expression, and then update memory_dict and memory_list
    :param memory: Memory Instance
    :param expression: [return, func, args..]
    :return:
    """
    return_value = expression[0]
    func_name = expression[1]
    func_def = FUNCTIONS[func_name]
    args_num = len(func_def["arguments"])
    func = func_def["func"]
    func_type = func_def["func_type"]

    if args_num == 1:
        argument = memory.read(expression[2])
        result = func(argument.value)
        memory_entry = MemoryEntry(
            value=result,
            name=return_value,
            src1=argument,
            opt=func_name,
            data_type=func_def["return_type"]
        )
    elif args_num == 2:
        if func_type == FunctionType.HIGHER_ORDER:
            lambda_expr = FUNCTIONS[expression[2]]["func"]
            argument = memory.read(expression[3])
            result = func(lambda_expr, argument.value)
            memory_entry = MemoryEntry(
                value=result,
                name=return_value,
                src1=argument,
                opt=func_name,
                data_type=func_def["return_type"]
            )
        else:
            argument1 = memory.read(expression[2])
            argument2 = memory.read(expression[3])
            result = func(argument1.value, argument2.value)
            memory_entry = MemoryEntry(
                value=result,
                name=return_value,
                src1=argument1,
                src2=argument2,
                opt=func_name,
                data_type=func_def["return_type"]
            )
    else:
        if func_type == FunctionType.HIGHER_ORDER:
            lambda_expr = FUNCTIONS[expression[2]]["func"]
            argument1 = memory.read(expression[3])
            argument2 = memory.read(expression[4])
            result = func(lambda_expr, argument1.value, argument2.value)
            memory_entry = MemoryEntry(
                value=result,
                name=return_value,
                src1=argument1,
                src2=argument2,
                opt=func_name,
                data_type=func_def["return_type"]
            )
        else:
            argument1 = memory.read(expression[2])
            argument2 = memory.read(expression[3])
            argument3 = memory.read(expression[4])
            result = func(argument1.value, argument2.value, argument3.value)
            memory_entry = MemoryEntry(
                value=result,
                name=return_value,
                src1=argument1,
                src2=argument2,
                src3=argument3,
                opt=func_name,
                data_type=func_def["return_type"]
            )

    if memory_entry.value is None:
        raise Exception("Output is None")

    # Check value
    if memory_entry.type == DataType.INT:
        # INT
        if (memory_entry.value < ps.MIN_VALUE) or (memory_entry.value > ps.MAX_VALUE):
            raise Exception("Value out of range")
    else:
        # INT LIST
        if len(memory_entry.value) > ps.INT_LIST_MAX_LENGTH:
            raise Exception("INT LIST out of range")

        for v in memory_entry.value:
            if v < ps.MIN_VALUE or v > ps.MAX_VALUE:
                raise Exception("INT LIST Value out of range")

    memory.write(return_value, memory_entry)


def run(inputs, expressions, output_variable_name):
    """
    Run program
    :param inputs:
    :param expressions: program expression
    :param output_variable_name:
    :return:
        Memory: Memory Instance
        Output: MemoryEntry Instance
    """
    memory = Memory()
    for i in inputs:
        entry = MemoryEntry(
            name=i["variable_name"],
            value=i["value"],
            data_type=i["data_type"],
            opt=None
        )
        memory.write(i["variable_name"], entry)
    for expression in expressions:
        interpret(memory, expression)

    return memory, memory.read(output_variable_name)


if __name__ == "__main__":

    _inputs = [{
        "variable_name": "a",
        "value": [1, 2, 3, 4, 5],
        "data_type": DataType.INT_LIST
    }]

    _expressions = [['b', 'REVERSE', 'a'], ['c', 'SCAN1', 'MAX', 'b']]
    _output_variable_name = "c"

    _memory, _output = run(_inputs, _expressions, _output_variable_name)
    print(_memory)
    print(_output)
