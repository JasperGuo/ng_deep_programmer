# coding=utf8

import sys

sys.path += ".."

import os
import json
import argparse
import interpreter
from program import Program, TestCase
from memory import Memory, MemoryEntry
from generate import generate_program_input
from program_generator.func_set import DataType


TEST_CASE_NUM = 10
TEST_CASE_MAX_RETRY = 1000


def main(program_dict_file, program_str_file):

    with open(program_dict_file, "r") as f:
        program_dicts = json.load(f)

    with open(program_str_file, "r") as f:
        program_strs = set(json.load(f))

    error = list()
    correct = list()
    for program_dict in program_dicts:
        program = Program.deserialize(program_dict)

        for test_case in program.test_cases:
            memory = Memory()
            inputs = test_case.inputs
            for i in inputs:
                entry = MemoryEntry(
                    name=i["variable_name"],
                    value=i["value"],
                    data_type=i["data_type"],
                    opt=None
                )
                memory.write(i["variable_name"], entry)
            expressions = program.expressions()
            try:
                for expression in expressions:
                    interpreter.interpret(memory, expression)
            except Exception as e:
                print(e)
                break
            else:
                output = memory.read(program.output()["variable_name"])
                test_case.output.update({
                    "value": output.value
                })
        else:
            correct.append(program.serialize())
            continue

        # One of the test case fail, generate another
        program.test_cases = list()
        for i in range(TEST_CASE_NUM):
            v = 0
            while v < TEST_CASE_MAX_RETRY:
                inputs = generate_program_input(program)
                try:
                    memory, output = interpreter.run(inputs, program.expressions(), program.output()["variable_name"])
                    if output.type == DataType.INT_LIST and len(output.value) == 0:
                        raise Exception("Empty List")
                except Exception as e:
                    v += 1
                else:
                    test_case = TestCase(inputs, {
                        "variable_name": output.name,
                        "value": output.value,
                        "data_type": output.type
                    })
                    program.add_test_case(test_case)
                    break
            else:
                break
        else:
            correct.append(program.serialize())
            continue

        # Fail
        program_strs -= {program.to_string()}
        error.append(program.serialize())

    with open(program_dict_file, "w") as f:
        f.write(json.dumps(correct))

    with open(program_str_file, "w") as f:
        f.write(json.dumps(list(program_strs)))

    print("Error: %d" % len(error))


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--programs", help="Program Dict file", required=True)
    args_parser.add_argument("--programset", help="Program Str file", required=True)

    args = args_parser.parse_args()

    main(args.programs, args.programset)

