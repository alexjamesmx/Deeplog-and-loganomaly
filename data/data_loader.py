import os
import json
import logging
import pandas as pd
import yaml
import argparse
from logging import Logger, getLogger
from data.window import sliding_window
from collections import Counter

import pickle
from accelerate import Accelerator
from utils.helpers import arg_parser


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

accelerator = Accelerator()


def process_dataset(logger: Logger,
                    output_dir: str,
                    args: argparse.Namespace
                    ):
    """
    Convert and read txt file to json 
    Create windowed sessions from json file
    Split windows to train and test
    Create or load train and test pkl files from txt file

    Args:
        logger (Logger)
        output_dir (str) : output directory to save train and test pkl files 
        args (argparse.Namespace)

    Raises:
        NotImplementedError: File does not exist
        NotImplementedError: Grouping method is not implemented

    Returns:
        _type_: train and test pkl file paths
    """

    # load train, test pkl's if exists "./output/.../train.pkl", "./output/.../test.pkl" and "./dataset/{dataset_name}/data.json"
    if os.path.exists(os.path.join(output_dir, "train.pkl")) and os.path.exists(os.path.join(output_dir, "test.pkl")) and os.path.exists(os.path.join(args.data_dir+args.dataset_folder, "data.json")):
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

        json_objects, count_errors = load_json_objects(
            txt_file_path, num_lines=100000)
        # count keys optional for debugging
        count_keys = count_json_keys(json_objects)
        logger.info(
            f"Total logs in {txt_file_path}: {len(json_objects)}, errors {count_errors}")
        print(count_keys)
        df = pd.DataFrame(json_objects)
        # save txt to json
        df.to_json(json_file_path, orient='records', lines=True)

    if args.grouping == "sliding":
        window_df = sliding_window(
            df=df, window_size=args.window_size, step_size=args.step_size,
            logger=logger)
        n_train = int(args.train_size * len(window_df))
        train_window = window_df[:n_train]
        test_window = window_df[n_train:]
        logger.info(
            f"train sessions: {len(train_window)}, test sessions: {len(test_window)} train_size: {args.train_size} len(window_df): {len(window_df)}")
    else:
        raise NotImplementedError(
            f"{args.grouping} grouping method is not implemented")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.pkl"), mode="wb") as f:
        pickle.dump(train_window, f)
    with open(os.path.join(output_dir, "test.pkl"), mode="wb") as f:
        pickle.dump(test_window, f)
    logger.info(
        f"Saved train.pkl and test.pkl at {output_dir}")
    return os.path.join(output_dir, "train.pkl"), os.path.join(output_dir, "test.pkl")


def load_json_objects(data_dir, num_lines=None):
    json_objects = []
    count_errors = 0
    line_count = 0
    with open(data_dir, 'r') as f:
        for line in f:
            if num_lines is not None and line_count >= num_lines:
                break
            try:
                json_obj = json.loads(line)
                if isinstance(json_obj, dict):
                    json_objects.append(json_obj)
                    line_count += 1
                else:
                    # print(f"Error: {line}")
                    count_errors += 1
            except json.JSONDecodeError as e:
                # print(e)
                count_errors += 1
    return json_objects, count_errors


def count_json_keys(json_objects):
    count_keys = Counter()
    for log in json_objects:
        count_keys.update(log.keys())
    return count_keys


# run as python -m data.data_loader
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

    logger = getLogger(args.model_name)
    logger.info(accelerator.state)
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    output_dir = f"{args.output_dir}{args.model_name}/train{args.train_size}/w_size{args.window_size}_s_size{args.step_size}/"

    process_dataset(logger=logger,
                    output_dir=output_dir,
                    args=args
                    )
