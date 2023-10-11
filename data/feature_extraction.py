from collections import Counter
import pickle
from tqdm import tqdm
from typing import List, Tuple, Optional, Any
import numpy as np
import sys
from data.store import Store


def load_features(data_path=str, min_len=0, is_train=True, store=Store):
    """
    Load features from pickle file
    and convert list of dicts to list of tuples

    Parameters
    ----------
    data_path: str: Path to pickle file
    min_len: int: Minimum length of log sequence
    pad_token: str: Padding token
    is_train: bool: Whether the data is training data or not
    store: Store
    Returns
    -------
    logs: List[Tuple[int, List[str], List[str]]]: List of log sequences
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    if is_train:
        store.set_train_data(data)
    else:
        print("setting original data")
        store.set_test_data(data)

    logs = []
    for seq in data:
        # print(f"seq: {seq['SEVERITY']} \n")
        if isinstance(seq["EventId"], list) and len(seq['EventId']) < min_len:
            continue

        logs.append(
            (seq["SessionId"], seq['EventId'], seq['SEVERITY'], seq["_zl_timestamp"], seq["log_uuid"]))

    # length is the length of each log sequence (window type) at position 1 (list of events), where each log is an array of [sessionId, eventId, label].
    logs_len = [len(log[2]) for log in logs]
    return logs, {"min": min(logs_len), "max": max(logs_len), "mean": np.mean(logs_len)}


def sliding_window(data: List[Tuple[List[str], int]],
                   window_size: int = 40,
                   is_train: bool = True,
                   parameter_model: bool = False,
                   vocab: Optional[Any] = None,
                   sequential: bool = False,
                   quantitative: bool = False,
                   semantic: bool = False,
                   logger: Optional[Any] = None,
                   ) -> Any:
    """
    Sliding window for log sequence
    Parameters
    ----------
    data: List[Tuple[List[str], int]]: List of log sequences
    window_size: int: Size of sliding window
    is_train: bool: training mode or not
    vocab: Optional[Any]: Vocabulary
    sequential: bool: Whether to use sequential features
    quantitative: bool: Whether to use quantitative features
    semantic: bool: Whether to use semantic features
    logger: Optional[Any]: Logger

    Returns
    -------
    lists of sequential, quantitative, semantic features, and labels
    """
    log_sequences = []

    if is_train:

        sessionIds = {}
        for idx, (sessionId, eventIds, severity, timestamp, *log_uuid) in tqdm(enumerate(data), total=len(data),
                                                                               desc=f"Train - Sliding window with size {window_size}"):
            # print(
            #     f"sessionId: {sessionId}  eventIds: {eventIds}  severity: {severity}  timestamp: {timestamp}  log_uuid: {log_uuid} \n")

            # event_ids_list = list(eventIds)
            # add pading if the length of the line is less than the window size (when window = n, but last window has k < n  events, add n - k  padding events)
            eventIds = eventIds + \
                [vocab.pad_token] * (window_size - len(eventIds) + 1)

            timestamp_list = list(timestamp)

            sessionIds[idx] = (sessionId)
            for i in range(len(eventIds) - window_size):
                # get the index of the event in the window sequence
                # print(f"total events: {(eventIds)}")
                # print(
                # f"TEST: i + window_size({window_size})= {i + window_size} ")
                label = vocab.get_event(eventIds[i + window_size],
                                        use_similar=quantitative)  # use_similar only for LogAnomal
                # print(f"label i = {i + window_size} label result: {label}\n")

                seq = eventIds[i: i + window_size]
                # print(f"sequence i = {i}: {i + window_size} \n")
                sequential_pattern = [vocab.get_event(
                    event, use_similar=quantitative) for event in seq]

                sequence = {'nSec': i, 'sequential': sequential_pattern}
                sequence['label'] = label
                sequence['idx'] = idx
                # print("sequence: ", sequence, '\n')

                # build parameters value matrix
                if parameter_model:
                    current_timestamp = int(timestamp_list[i])
                    last_timestamp = 0
                    if i != 0:
                        last_timestamp = int(timestamp_list[i-1])
                    t_difference = current_timestamp - last_timestamp
                    sequence["parameters_values"] = np.array(
                        [t_difference, severity[i]])

                # print(
                #     f"tcurrent_t_last: {tcurrent_t_last}: {current_timestamp} - {last_timestamp} \n")
                log_sequences.append(sequence)
        sequentials, quantitatives, semantics = None, None, None
        if sequential:
            sequentials = [seq['sequential'] for seq in log_sequences]
            steps = [seq['nSec'] for seq in log_sequences]
        if quantitative:
            quantitatives = [seq['quantitative'] for seq in log_sequences]
        if semantic:
            print("semantic \n")
            semantics = [seq['semantic'] for seq in log_sequences]
        if parameter_model:
            parameters = [seq['parameters_values']
                          for seq in log_sequences]
        labels = [seq['label'] for seq in log_sequences]
        sequence_idxs = [seq['idx'] for seq in log_sequences]
        logger.info(f"Number of sequences: {len(labels)}")

        return sequentials, quantitatives, semantics, labels, sequence_idxs, sessionIds, parameters, steps
