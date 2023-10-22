import os
import json
import pandas as pd
import yaml
import pickle
import argparse

from typing import Tuple
from collections import Counter
from logging import getLogger, Logger

from data.window import sliding_window
from utils.helpers import arg_parser


def process_dataset(args: argparse.Namespace,
                    output_dir: str,
                    logger: Logger = getLogger("__name__")) -> Tuple[str, str]:
    """
    1-Converts and read txt files to json to handle them easily  
    2-Creates windowed sessions from json file
    3-Splits windows to train and test
    4-Creates or loads train and test .pkl files 
    Args:
        args (argparse.Namespace)
        output_dir (str): output directory
        logger (Logger)
    Raises:
        NotImplementedError: Data file does not exist
        NotImplementedError: Grouping method is not implemented (for now only sliding is implemented)
    Returns:
        _type_: train and test pkl file paths
    """

    # load train / test .pkl files if exists
    # e.g "./output/.../train.pkl and ./output/.../test.pkl and ./dataset/{dataset_name}/data.json"
    if os.path.exists(os.path.join(output_dir, "train.pkl")) and os.path.exists(os.path.join(output_dir, "test.pkl")) and os.path.exists(os.path.join(args.data_dir+args.dataset_folder, args.log_file + ".json")):
        logger.info(
            f"Loading {output_dir}/train.pkl and {output_dir}/test.pkl")
        return os.path.join(output_dir, "train.pkl"), os.path.join(output_dir, "test.pkl")
    # load json if exists  ./dataset/{dataset_name}/{log_file}.json
    json_file_path = f"{args.data_dir}{args.dataset_folder}{args.log_file}.json"
    txt_file_path = f"{args.data_dir}{args.dataset_folder}{args.log_file}.txt"
    if os.path.isfile(json_file_path):
        logger.info(f"Loading {json_file_path}")
        df = pd.read_json(json_file_path, orient='records', lines=True)
    else:
        if not os.path.isfile(txt_file_path):
            raise NotImplementedError(
                f"The file {txt_file_path}.txt does not exist")

        logs, count_errors = raw_logs_tojson(
            txt_file_path, num_lines=100000)
        # count all keys (uncomment for debugging)
        total_keys = count_keys(logs)
        logger.info(
            f"Total logs in {txt_file_path}: {len(logs)}, errors {count_errors}")
        print("Total keys: ", total_keys)
        df = pd.DataFrame(logs)
        df.to_json(json_file_path, orient='records', lines=True)
    # Create sessions
    if args.grouping == "sliding":
        window_df = sliding_window(df,
                                   window_size=args.window_size,
                                   step_size=args.step_size,
                                   logger=logger)

        n_train = int(args.train_size * len(window_df))
        train_window = window_df[:n_train]
        test_window = window_df[n_train:]
        logger.info(
            f"Train sessions: {len(train_window)} | Test sessions: {len(test_window)} | Train_size: {args.train_size}% | Total sessions: {len(window_df)}")
    else:
        raise NotImplementedError(
            f"{args.grouping} grouping method is not implemented")
    # Save sessions to .pkl files
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.pkl"), mode="wb") as f:
        pickle.dump(train_window, f)
    with open(os.path.join(output_dir, "test.pkl"), mode="wb") as f:
        pickle.dump(test_window, f)
    logger.info(
        f"Saved train.pkl and test.pkl at {output_dir}")
    return os.path.join(output_dir, "train.pkl"), os.path.join(output_dir, "test.pkl")


def raw_logs_tojson(data_dir, num_lines=None):
    """
    Parses txt rows to json objects
    Discards rows that are not json objects

    Args:
        data_dir (str): path to txt file  
        num_lines (_type_, optional): No. json objects to save (testing). Defaults to None.

    Returns:
        Tuple[List[dict], int]: List of json objects and number of errors 
    """
    json_objects = []
    count_errors = 0
    line_count = 0
    no_eventId = 0
    with open(data_dir, 'r') as f:
        for line in f:
            if num_lines is not None and line_count >= num_lines:
                break
            try:
                json_obj = json.loads(line)
                if isinstance(json_obj, dict):
                    # print(json_obj)
                    if json_obj.get("EVENTID") is None:
                        # skip if no eventId
                        no_eventId += 1
                        # print(f"Useless: {json_obj}")
                    else:
                        json_objects.append(json_obj)
                        line_count += 1
                else:
                    print(f"Error: {line}")
                    count_errors += 1
            except json.JSONDecodeError as e:
                print(e)
                count_errors += 1
        print(f"total no eventIds: {no_eventId}")
    return json_objects, count_errors


def count_keys(logs):
    """
    This is used for testing. 
    Counts all keys from dataset.

    Note:
        Not all logs have the same keys (unknown keys might appear), making it hard to train.
        Deeplog uses eventTemplates, however, our data does not have this key so we use evenIDs
        Not every logs have a eventId (currently causing trouble).
    Args:
        logs (List[dict]): 

    Returns:
        Counter: Counter of all keys
    """
    count_keys = Counter()
    for log in logs:
        count_keys.update(log.keys())
    return count_keys


# run as python -m data.data_loader
# debugging purposes
if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()

    if args.config_file is not None and os.path.isfile(args.config_file):
        config_file = args.config_file
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config_args = argparse.Namespace(**config)
            for k, v in config_args.__dict__.items():
                if v is not None:
                    setattr(args, k, v)
        print(f"Loaded config from {config_file}!")
    else:
        print(f"Loaded config from command line!")

    output_dir = f"{args.output_dir}{args.model_name}/train{args.train_size}/w_size{args.window_size}_s_size{args.step_size}/"

    process_dataset(logger=None,
                    output_dir=output_dir,
                    args=args
                    )
