# coding=utf8

import argparse
import json
import random
import program_settings as ps
import interpreter
from program_generator.func_set import DataType
from program_generator.build_graph import generate_program
from program import Program, TestCase


TEST_CASE_MAX_RETRY = 100
TEST_CASE_NUM = 10


def generate_program_input(program):
    """
    :param program: Program Instance
    :return:
    """
    func_set = program.funcs()
    inputs = program.inputs()

    if "(**2)" in func_set:
        max_value = 15
        min_value = -15
    else:
        max_value = ps.MAX_VALUE
        min_value = ps.MIN_VALUE

    for i in inputs:
        data_type = i["data_type"]
        if data_type == DataType.INT:
            value = random.randint(min_value, max_value)
        else:
            length = random.randint(1, 20)
            value = list()
            for _ in range(length):
                value.append(random.randint(min_value, max_value))
        i.update({
            "value": value
        })
    return inputs


def main(num_input, length, num, save_path):
    """
    :param num_input: Number of input nodes
    :param length:    Length of the program
    :param num:       Number of programs
    :param save_path:
    :return:
    """
    program_set = set()
    curr = 0
    retry = 0
    programs = list()
    max_retry = num * 100
    while curr < num and retry < max_retry:
        try:
            data_flow_graph, tree, topological_sort_result = generate_program(num_input, length-1)
            p = Program(data_flow_graph, tree, topological_sort_result)
        except Exception as e:
            print(e)
            retry += 1
            continue
        program_string = p.to_string()
        expressions = p.expressions()
        if len(expressions) != length or program_string in program_set:
            retry += 1
        else:
            output_vertex = p.output()
            for i in range(TEST_CASE_NUM):
                v = 0
                while v < TEST_CASE_MAX_RETRY:
                    inputs = generate_program_input(p)
                    try:
                        memory, output = interpreter.run(inputs, p.expressions(), output_vertex["variable_name"])
                        if output.type == DataType.INT_LIST and len(output.value) == 0:
                            raise Exception("Empty List")
                    except Exception as e:
                        # print(e)
                        v += 1
                    else:
                        test_case = TestCase(inputs, {
                            "variable_name": output.name,
                            "value": output.value,
                            "data_type": output.type
                        })
                        p.add_test_case(test_case)
                        break
                else:
                    break
            else:
                program_set.add(program_string)
                programs.append(p.serialize())
                retry = 0
                curr += 1
            retry += 1

    if save_path:
        print("Num: %d" % len(programs))
        with open(save_path, "w") as f:
            f.write(json.dumps(programs, indent=4))
    else:
        print("Done: ")
        for p_str in programs:
            print(p_str)
            program = Program.deserialize(p_str)
            program.print()
            print(program.inputs())
            print(program.expressions())
            print(program.tree)
            print(program.output())
            print("===============================")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input", help="Number of input", required=True)
    arg_parser.add_argument("--length", help="Length of program", required=True)
    arg_parser.add_argument("--num", help="Number of program", required=True)
    arg_parser.add_argument("--save", help="Save path", required=False)

    args = arg_parser.parse_args()
    main(int(args.input), int(args.length), int(args.num), args.save)
