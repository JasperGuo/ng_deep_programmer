# coding=utf8

"""
Split training set, test set, and dev set
"""
import os
import json
import random
import argparse


def main(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    total = len(data)

    training_data_size = int(len(data) * 0.75)
    testing_data_size = int(len(data) * 0.15)
    dev_data_size = total - training_data_size - testing_data_size

    random.shuffle(data)

    training_data = data[:training_data_size]
    testing_data = data[training_data_size:training_data_size+testing_data_size]
    dev_data = data[training_data_size+testing_data_size:]

    path, filename = os.path.split(file_path)
    _, ext = os.path.splitext(filename)

    training_set_filename = os.path.join(path, _+"_train"+".json")
    testing_set_filename = os.path.join(path, _+"_test"+".json")
    dev_set_filename = os.path.join(path, _+"_dev"+".json")

    with open(training_set_filename, "w") as f:
        f.write(json.dumps(training_data))
    with open(testing_set_filename, "w") as f:
        f.write(json.dumps(testing_data))
    with open(dev_set_filename, "w") as f:
        f.write(json.dumps(dev_data))

    print("Training_size: %d" % len(training_data))
    print("Testing_size: %d" % len(testing_data))
    print("Dev_size: %d" % len(dev_data))


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--file", help="Program file", required=True)
    args = args_parser.parse_args()
    main(args.file)
