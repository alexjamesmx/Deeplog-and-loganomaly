import os
import pandas as pd
import json
from collections import Counter
from typing import List, Tuple


def load_data_from_files(data_path: str) -> pd.DataFrame:
    """
    Load or create json file from txt file 

    Raises:
        NotImplementedError: If txt file path does not exist
    Returns: 
        df: dataframe of json file
    """
    json_path = f"{data_path}.json"
    txt_path = f"{data_path}.txt"

    if os.path.isfile(json_path):
        df = pd.read_json(json_path, orient="records", lines=True)
    else:
        if not os.path.isfile(txt_path):
            raise NotImplementedError(
                f"The file {txt_path}.txt does not exist")
        print("Creating json file from txt file...")
        logs, count_errors = raw_logs_tojson(txt_path, num_lines=100000)
        count_keys(logs)  # Comment this line if you dont want to count keys
        print(
            f"Total logs in {txt_path}: {len(logs)}, errors {count_errors}"
        )
        df = pd.DataFrame(logs)
        print("Saving json in ", json_path)
        df.to_json(json_path, orient="records", lines=True)
    return df


def raw_logs_tojson(data_dir, num_lines=None) -> Tuple[List[dict], int]:
    """
    Description:
        Parses txt log rows to json objects.
        Discards rows that are not in json format.

    Parameters:
        data_dir (str): path to txt file
        num_lines (int, optional): number of lines to read.

    Returns:
        Tuple[List[dict], int]: List of json objects and number of parsing errors
    """
    json_objects = []
    count_errors = 0
    line_count = 0
    no_eventId = 0
    with open(data_dir, "r") as f:
        for line in f:
            if num_lines is not None and line_count >= num_lines:
                break
            try:
                json_obj = json.loads(line)
                if isinstance(json_obj, dict):
                    if json_obj.get("EVENTID") is None:
                        # skip if no eventId
                        no_eventId += 1
                    else:
                        json_objects.append(json_obj)
                        line_count += 1
                else:
                    print(f"Error: {line}")
                    count_errors += 1
            except json.JSONDecodeError as e:
                print(e)
                count_errors += 1
        print(f"Total no eventIds: {no_eventId}")
    return json_objects, count_errors


def count_keys(logs: List[dict]):
    """(debugging)
    Description:
        Counts keys from dataset.

    Note:
        Not all logs have the same keys (unknown keys might appear).
        Deeplog/LogAnomaly use eventTemplates, however, our data does not have this key so we use evenIDs
        Not every logs have a eventId (causing trouble).

    Parameters:
        logs (List[dict]):

    """
    counter = Counter()
    for log in logs:
        counter.update(log.keys())
    print("Total keys: ", counter + "\n")
