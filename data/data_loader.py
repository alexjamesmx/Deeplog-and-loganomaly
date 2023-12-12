import os
import json
import pandas as pd
import yaml
import pickle
import argparse

from typing import Tuple, List
from collections import Counter
from logging import getLogger, Logger

from data.window import create_sliding_windows
from utils.helpers import arg_parser
from utils.common import load_data_from_files


def process_dataset(
    args: argparse.Namespace, output_dir: str, logger: Logger
) -> Tuple[str, str]:
    """
    Load train/text df (windowed sessions).

    Parameters:
        args (argparse.Namespace).
        output_dir (str): directory where .pkls are saved.
        logger (Logger).

    Raises:
        NotImplementedError: Data file does not exist.
        NotImplementedError: Grouping method is not implemented (for now only sliding is implemented).

    Returns:
        Tuple[str, str]: train.pkl and test.pkl paths.
    """
    train_path = os.path.join(output_dir, "train.pkl")
    test_path = os.path.join(output_dir, "test.pkl")

    if os.path.exists(train_path) and os.path.exists(test_path):
        logger.info(
            f"Loading {output_dir}/train.pkl and {output_dir}/test.pkl")
        return os.path.join(output_dir, "train.pkl"), os.path.join(
            output_dir, "test.pkl"
        )

    data_path = f"{args.data_dir}{args.dataset_folder}{args.log_file}"

    df = load_data_from_files(data_path)
    return df_to_windowed_sessions(df, output_dir, args, logger)


def df_to_windowed_sessions(
    df: pd.DataFrame,
    output_dir: str,
    args: argparse.Namespace,
    logger: Logger = getLogger("__model__"),
) -> Tuple[str, str]:
    """
    Description:
        Creates windowed sessions from df.

    Parameters:
        args (argparse.Namespace).
        df (pd.DataFrame).
        output_dir (str): directory where .pkls are saved.
        logger (Logger).

    Raises:
        NotImplementedError: _description_

    Returns:
        Tuple[str, str]: train.pkl and test.pkl paths.
    """
    if args.grouping == "sliding":
        window_df = create_sliding_windows(
            df, window_size=args.window_size, step_size=args.step_size, logger=logger
        )

        n_train = int(args.train_size * len(window_df))
        train_window = window_df[:n_train]
        test_window = window_df[n_train:]
        logger.info(
            f"Train sessions: {len(train_window)} | Test sessions: {len(test_window)} | Train_size: {args.train_size}% | Total sessions: {len(window_df)}"
        )
    else:
        raise NotImplementedError(
            f"{args.grouping} grouping method is not implemented")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.pkl"), mode="wb") as f:
        pickle.dump(train_window, f)
    with open(os.path.join(output_dir, "test.pkl"), mode="wb") as f:
        pickle.dump(test_window, f)
    logger.info(f"Saved train.pkl and test.pkl at {output_dir}")
    return os.path.join(output_dir, "train.pkl"), os.path.join(output_dir, "test.pkl")


# (debugging)
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

    process_dataset(logger=None, output_dir=output_dir, args=args)
