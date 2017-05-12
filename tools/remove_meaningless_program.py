# coding=utf8

"""
Remove Redundant program
"""

import sys

sys.path += ".."

import os
import json
import argparse
from program import Program


def find_redundant_funcs(program, vertex):
    """
    Find redundant funcs,
    For example:
        b = sort(a), c = sort(b)
        b = reverse(a), d = reverse(b), then a = d
    :param program:
    :param vertex:
    :return:
    """
    children = program.graph.children(vertex)
    vertex_func = program.graph.get_attr(vertex, "func")
    vertex_lambda = program.graph.get_attr(vertex, "lambda")
    for child in children:

        child_func = program.graph.get_attr(child, "func")
        child_lambda = program.graph.get_attr(child, "lambda")

        if vertex_func == child_func:
            if (vertex_func == "SORT") or (vertex_func == "REVERSE"):
                return True
            if ((vertex_func == "FILTER") or (vertex_func == "COUNT")) and (child_lambda == vertex_lambda):
                return True

        is_found = find_redundant_funcs(program, child)

        if is_found:
            return True

    return False


def check(program):
    """
    If redundant functions are found, return False, else return True
    :param program:
    :return:
    """
    inputs = program.inputs()
    for _input in inputs:
        is_found = find_redundant_funcs(program, _input["vertex"])
        if is_found:
            return False
    else:
        return True


def main(file, save_path):
    with open(file, "r") as f:
        program_dicts = json.load(f)

    redundant_count = 0
    correct = list()
    for program_dict in program_dicts:
        program = Program.deserialize(program_dict)

        # traverse program graph

        if check(program):
            # Pass
            correct.append(program_dict)
        else:
            redundant_count += 1
    print("Invalid Programs: %d" % redundant_count)

    if not save_path:
        _, _file = os.path.split(file)
        save_path = os.path.join(_, "ng_" + _file)

    with open(save_path, "w") as f:
        f.write(json.dumps(correct))

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--file", help="Program file", required=True)
    args_parser.add_argument("--save", help="Save path", required=False)

    args = args_parser.parse_args()

    main(args.file, args.save)

