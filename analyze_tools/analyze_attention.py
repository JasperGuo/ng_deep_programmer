# coding=utf8

"""
Analyze Attention
"""

import re
import argparse
from pprint import pprint

PREDICTED_OPT_PATTERN = re.compile(r"Predicted Opt: (.*)")
TRUTH_OPT_PATTERN = re.compile(r"Truth Opt: (.*)")
MEMORY_SIZE = re.compile(r"Memory Size: (\d+)")
MEMORY = re.compile(r"Memory:(.*)")
ARGS_PATTERN = re.compile(r"args:\s*(\[.*\])")
MEMORY_ENTRY_PATTERN = re.compile(r"(\d):\s+(0\.[0-9]+),\s+(int|int\[\]|<P>),\s+(\[.*\])\s*")
CASE_SPLIT_MATCH = re.compile(r"(\*)+")
SAMPLE_END_PATTERN = re.compile(r"(=)+")


def read(file):
    """
    Read File
    :param file:
    :return:
    """
    result = list()
    with open(file, "r") as f:
        line = f.readline()
        while line and line != "":
            match = PREDICTED_OPT_PATTERN.match(line)
            if match:
                predicted_opt = match.group(1).strip()
                truth_opt = TRUTH_OPT_PATTERN.match(f.readline()).group(1).strip()
                args = eval(ARGS_PATTERN.match(f.readline()).group(1).strip())
                line = f.readline()
                _m = MEMORY_SIZE.match(line)
                while not _m:
                    line = f.readline()
                    _m = MEMORY_SIZE.match(line)
                memory_size = int(_m.group(1).strip())

                line = f.readline()
                _m = MEMORY.match(line)
                while not _m:
                    line = f.readline()
                    _m = MEMORY.match(line)

                # Begin to Look Case, Memory Entry
                line = f.readline()
                _m = SAMPLE_END_PATTERN.match(line)
                while not _m:
                    # Read Memory Entry

                    is_case_end = CASE_SPLIT_MATCH.match(line)
                    while not is_case_end:
                        line = f.readline()
                        entry_match = MEMORY_ENTRY_PATTERN.match()

                r = {
                    "predicted_opt": predicted_opt,
                    "truth_opt": truth_opt,
                    "memory_size": memory_size,
                    "args": args
                }
                result.append(r)

            line = f.readline()
    return result
