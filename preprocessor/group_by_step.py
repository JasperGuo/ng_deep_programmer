# coding=utf8

import sys
sys.path += ".."

import re
import os
import uuid
import json
import argparse
import interpreter
from program import Program
from memory import Memory, MemoryEntry


MAX_LENGTH = 10
PROGRAM_FILENAME_PATTERN = re.compile(r'(\d+)_input.json')


def process(program):

    pid = str(uuid.uuid1())

    if not isinstance(program, Program):
        program = Program.deserialize(program)

    program_length = program.length
    program_trace = {str(n): [] for n in range(program_length+1)}
    for tid, test_case in enumerate(program.test_cases):
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

        idx = 0
        while idx < len(expressions):
            expression = expressions[idx]
            curr_trace = str(program_length - idx)
            program_trace[curr_trace].append({
                "test_case_id": tid,
                "step": int(curr_trace),
                "output": test_case.output,
                "func": expression[1],
                "args": expression[2:],
                "memory": memory.serialize()
            })
            interpreter.interpret(memory, expression)
            idx += 1
        program_trace["0"].append({
            "test_case_id": tid,
            "step": 0,
            "output": test_case.output,
            "func": None,
            "args": [],
            "memory": memory.serialize()
        })

    result = dict()
    for key, value in program_trace.items():
        result[key] = {
            "program_id": pid,
            "program_length": program.length,
            "detail": value
        }

    return result


def main(directory, save_path, save_base_name):

    traces = {str(n): [] for n in range(MAX_LENGTH + 1)}
    for file in os.listdir(directory):

        if not PROGRAM_FILENAME_PATTERN.match(file):
            continue

        with open(os.path.join(directory, file), "r") as f:
            programs = json.load(f)

        for program in programs:
            p_trace = process(program)
            for key, value in p_trace.items():
                traces[key].append(value)

    # Save
    for key, value in traces.items():
        path = os.path.join(save_path, save_base_name + "_" + key + ".json")
        basepath = os.path.dirname(path)
        if not os.path.exists(basepath):
            os.mkdir(basepath)
        with open(path, "w") as f:
            f.write(json.dumps(value))


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--directory", help="Program file directory", required=True)
    args_parser.add_argument("--save", help="Save base path", required=True)
    args_parser.add_argument("--basename", help="Save base path", required=True)

    args = args_parser.parse_args()

    main(args.directory, args.save, args.basename)
