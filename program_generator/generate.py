# coding=utf8

import argparse
import json
import os
from build_graph import generate_program, Program


MAX_RETRY = 100


def main(num_input, length, num, save_path):
    """
    :param num_input: Number of input nodes
    :param length:    Length of the program
    :param num:       Number of programs
    :return:
    """
    program_set = set()
    curr = 0
    retry = 0
    programs = list()
    while curr < num and retry < MAX_RETRY:
        try:
            p = generate_program(num_input, length-1)
        except Exception as e:
            print(e)
            retry += 1
            continue
        program_string = p.to_string()
        expressions = p.expressions()
        if len(expressions) != length or program_string in program_set:
            retry += 1
        else:
            program_set.add(program_string)
            programs.append(p.serialize())
            retry = 0
            curr += 1

    if save_path:
        with open(save_path, "w") as f:
            f.write(json.dumps(programs, indent=4))
    else:
        for p_str in programs:
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
