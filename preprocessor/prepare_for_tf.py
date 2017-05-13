# coding=utf8

import sys
sys.path += ".."

import json
import argparse
from program_generator.func_set import FUNCTIONS


def index(memory, var_name):
    for idx, value in enumerate(memory):
        if value["variable_name"] == var_name:
            return idx
    return None


def process(step_file):
    result = list()
    with open(step_file, "r") as f:
        traces = json.load(f)
    for trace in traces:
        program_dict = {
            "program_id": trace["program_id"],
            "program_length": trace["program_length"],
            "detail": []
        }
        for test_case_run_detail in trace["detail"]:
            memory = test_case_run_detail["memory"]
            formatted_memory = list()
            for memory_entry in memory:
                for src in ["src1", "src2"]:
                    if memory_entry[src]:
                        idx = index(formatted_memory, memory_entry[src])
                        if idx is None:
                            raise Exception("Invalid Memory")
                        memory_entry[src] = idx
                formatted_memory.append(memory_entry)

            args = [None]*3
            for idx, arg in enumerate(test_case_run_detail["args"]):
                if arg in list(FUNCTIONS.keys()):
                    args[idx] = (arg, "FUNCTION")
                else:

                    memory_index = index(formatted_memory, arg)

                    if memory_index is None:
                        raise Exception("Invalid Memory")

                    args[idx] = (memory_index, "VAR")

            program_dict["detail"].append({
                "memory": formatted_memory,
                "step": test_case_run_detail["step"],
                "output": test_case_run_detail["output"],
                "test_case_id": test_case_run_detail["test_case_id"],
                "args": args,
                "func": test_case_run_detail["func"]
            })
        result.append(program_dict)
    return result


def main(files, save_path):
    result = list()
    for file in files:
        result += process(file.strip())

    with open(save_path, "w") as f:
        f.write(json.dumps(result))


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--files", help="Program files", required=True)
    args_parser.add_argument("--save", help="Save base path", required=True)

    args = args_parser.parse_args()

    _files = args.files.split(",")
    print(_files)
    main(_files, args.save)
