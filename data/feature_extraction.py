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
        if isinstance(seq["EventId"], list) and len(seq['EventId']) < min_len:
            continue
        logs.append(
            (seq["SessionId"], seq['EventId'], seq['Label']))

    # length is the length of each log sequence (window type) at position 1 (list of events), where each log is an array of [sessionId, eventId, label].
    logs_len = [len(log[1]) for log in logs]
    return logs, {"min": min(logs_len), "max": max(logs_len), "mean": np.mean(logs_len)}


def sliding_window(data: List[Tuple[List[str], int]],
                   window_size: int = 40,
                   is_train: bool = True,
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

    # print("vocab ", vocab.stoi)

    if is_train:

        sessionIds = {}
        for idx, (sessionId, eventIds, labels) in tqdm(enumerate(data), total=len(data),
                                                       desc=f"Train - Sliding window with size {window_size}"):
            # print(
            #     f"idx {idx} sessionId {sessionId} eventIds {eventIds} labels {labels} \n")
            # print(
            #     f"length of eventIds {len(eventIds)} \n len line - window_size {len(eventIds) - window_size} \n")

            line = list(eventIds)
            line = line + [vocab.pad_token] * (window_size - len(line) + 1)

            sessionIds[idx] = (sessionId)

            for i in range(len(line) - window_size):
                # get the index of the event in the window sequence
                label = vocab.get_event(line[i + window_size],
                                        use_similar=quantitative)  # use_similar only for LogAnomal
                # print(f"label i = {i + window_size} \n")

                seq = line[i: i + window_size]
                # print(f"sequence i = {i}: {i + window_size} \n")
                sequential_pattern = [vocab.get_event(
                    event, use_similar=quantitative) for event in seq]
                semantic_pattern = None
                if semantic:
                    print("semantic")
                    semantic_pattern = [
                        vocab.get_embedding(event) for event in seq]
                quantitative_pattern = None
                if quantitative:
                    print("quantitive")
                    quantitative_pattern = [0] * len(vocab)
                    log_counter = Counter(sequential_pattern)
                    for key in log_counter:
                        try:
                            quantitative_pattern[key] = log_counter[key]
                        except Exception as _:
                            pass  # ignore unseen events or padding key

                sequence = {'nSec': i, 'sequential': sequential_pattern}
                if quantitative:
                    sequence['quantitative'] = quantitative_pattern
                if semantic:
                    sequence['semantic'] = semantic_pattern
                sequence['label'] = label
                sequence['idx'] = idx
                log_sequences.append(sequence)
        sequentials, quantitatives, semantics = None, None, None
        if sequential:
            sequentials = [seq['sequential'] for seq in log_sequences]
        if quantitative:
            quantitatives = [seq['quantitative'] for seq in log_sequences]
        if semantic:
            print("semantic \n")
            semantics = [seq['semantic'] for seq in log_sequences]
        labels = [seq['label'] for seq in log_sequences]
        sequence_idxs = [seq['idx'] for seq in log_sequences]
        logger.info(f"Number of sequences: {len(labels)}")

        return sequentials, quantitatives, semantics, labels, sequence_idxs, sessionIds
