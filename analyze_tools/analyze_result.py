# coding=utf8

"""
Analyze Test Result
"""

import re
import argparse
from pprint import pprint

PREDICTED_OPT_PATTERN = re.compile(r"Predicted Opt: (.*)")
TRUTH_OPT_PATTERN = re.compile(r"Truth Opt: (.*)")
MEMORY_SIZE = re.compile(r"Memory Size: (\d+)")


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
                line = f.readline()
                _m = MEMORY_SIZE.match(line)
                while not _m:
                    line = f.readline()
                    _m = MEMORY_SIZE.match(line)
                memory_size = int(_m.group(1).strip())

                r = {
                    "predicted_opt": predicted_opt,
                    "truth_opt": truth_opt,
                    "memory_size": memory_size
                }
                result.append(r)

            line = f.readline()
    return result


def main(file):
    result = read(file)

    correct_dict = dict()
    error_dict = dict()
    total = 0
    correct_count = 0
    error_count = 0

    error_opt_count = dict()
    correct_opt_count = dict()

    for r in result:
        predicted_opt = r["predicted_opt"]
        truth_opt = r["truth_opt"]
        memory_size = r["memory_size"]

        total += 1

        if predicted_opt != truth_opt:
            _dict = error_dict
            error_count += 1

            if truth_opt not in error_opt_count:
                error_opt_count[truth_opt] = {
                    "detail": dict(),
                    "total": 0
                }
            if predicted_opt not in error_opt_count[truth_opt]["detail"]:
                error_opt_count[truth_opt]["detail"][predicted_opt] = 0
            error_opt_count[truth_opt]["detail"][predicted_opt] += 1
            error_opt_count[truth_opt]["total"] += 1
        else:
            _dict = correct_dict
            correct_count += 1

            if truth_opt not in correct_opt_count:
                correct_opt_count[truth_opt] = 0
            correct_opt_count[truth_opt] += 1

        if memory_size not in _dict:
            _dict[memory_size] = {
                "total": 0,
                "detail": dict()
            }
        if truth_opt not in _dict[memory_size]["detail"]:
            _dict[memory_size]["detail"][truth_opt] = 0
        _dict[memory_size]["detail"][truth_opt] += 1
        _dict[memory_size]["total"] += 1

    """
    for opt, value in error_opt_count.items():
        error_opt_count[opt] = round(float(value)/float(error_count), 2)
    """

    size_dict = dict()
    for (size, value) in error_dict.items():
        size_dict[size] = round(float(value["total"])/float(error_count), 2)

    print("Total: %d" % total)
    print("Correct Total: %d" % correct_count)
    print("Error Total:   %d" % error_count)
    print("\n")
    print("Correct: ")
    pprint(correct_dict)
    print("\n")
    print("Error: ")
    pprint(error_dict)

    print("\nError Opt: ")
    pprint(error_opt_count)

    print("\nCorrect Opt: ")
    pprint(correct_opt_count)

    print("\nError Memory Size: ")
    pprint(size_dict)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--file", help="Test log file path", required=True)

    args = arg_parser.parse_args()
    main(args.file)
