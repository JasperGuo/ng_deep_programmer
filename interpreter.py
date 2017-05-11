# coding=utf8

from program_generator.func_set import FUNCTIONS, FunctionType, DataType
from memory import MemoryEntry


def interpret(memory_dict, memory_list, expression):
    """
    interpret expression, and then update memory_dict and memory_list
    :param memory_dict:
    :param memory_list:
    :param expression: [return, func, args..]
    :return:
    """
    return_value = expression[0]
    func_def = FUNCTIONS[expression[1]]
    args_num = len(func_def["arguments"])
    func = func_def["func"]
    func_type = func_def["func_type"]

    if args_num == 1:
        argument = memory_dict[expression[2]]
        result = func(argument.value)
        memory_entry = MemoryEntry(
            value=result,
            name=return_value,
            src1=argument,
            opt=func,
            data_type=func_def["return_type"]
        )
    elif args_num == 2:
        if func_type == FunctionType.HIGHER_ORDER:
            lambda_expr = FUNCTIONS[expression[2]]["func"]
            argument = memory_dict[expression[3]]
            result = func(lambda_expr, argument.value)
            memory_entry = MemoryEntry(
                value=result,
                name=return_value,
                src1=argument,
                opt=func,
                data_type=func_def["return_type"]
            )
        else:
            argument1 = memory_dict[expression[2]]
            argument2 = memory_dict[expression[3]]
            result = func(argument1.value, argument2.value)
            memory_entry = MemoryEntry(
                value=result,
                name=return_value,
                src1=argument1,
                src2=argument2,
                opt=func,
                data_type=func_def["return_type"]
            )
    else:
        if func_type == FunctionType.HIGHER_ORDER:
            lambda_expr = FUNCTIONS[expression[2]]["func"]
            argument1 = memory_dict[expression[3]]
            argument2 = memory_dict[expression[4]]
            result = func(lambda_expr, argument1.value, argument2.value)
            memory_entry = MemoryEntry(
                value=result,
                name=return_value,
                src1=argument1,
                src2=argument2,
                opt=func,
                data_type=func_def["return_type"]
            )
        else:
            argument1 = memory_dict[expression[2]]
            argument2 = memory_dict[expression[3]]
            argument3 = memory_dict[expression[4]]
            result = func(argument1.value, argument2.value, argument3.value)
            memory_entry = MemoryEntry(
                value=result,
                name=return_value,
                src1=argument1,
                src2=argument2,
                src3=argument3,
                opt=func,
                data_type=func_def["return_type"]
            )
    memory_dict.update({
        return_value: memory_entry
    })
    memory_list.append(memory_entry)


if __name__ == "__main__":
    entry1 = MemoryEntry(name="a", value=[1, 2, 3], data_type=DataType.INT_LIST, opt=None)
    entry2 = MemoryEntry(name="b", value=[1, 2, 3], data_type=DataType.INT_LIST, opt=None)
    memory_dict = {
        "a": entry1,
        "b": entry2
    }
    memory_list = [entry1, entry2]
    expression = ["c", "COUNT", "(<0)", "a"]

    interpret(memory_dict, memory_list, expression)
    print(memory_dict)
    print(memory_list)
