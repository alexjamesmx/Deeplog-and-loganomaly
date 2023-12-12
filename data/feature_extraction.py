from collections import Counter
import pickle
from tqdm import tqdm
from typing import List, Tuple, Optional, Any
import numpy as np
from data.store import Store
from data.vocab import Vocab
from logging import getLogger, Logger


def load_features(
    data_path: str, min_len: int = 0, is_train: bool = True, store: Store = None
):
    """
    Description:
        Load features from pickle file and convert list of dicts to list of tuples

    Parameters:
        data_path: str: Path to pickle file
        min_len: int: Minimum length of log sequence
        pad_token: str: Padding token
        is_train: bool: Whether the data is training data or not
        store: Store

    Returns:
        logs: List[Tuple[int, List[int], List[str], List[str], List[str]]]: List of log sequences
        stat: dict: Statistics of log sequences
    """
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    if is_train:
        store.set_train_data(data)
    else:
        store.set_test_data(data)

    logs = []
    for seq in data:
        if isinstance(seq["EventId"], list) and len(seq["EventId"]) < min_len:
            continue

        logs.append(
            (
                seq["SessionId"],
                seq["EventId"],
                seq["SEVERITY"],
                seq["_zl_timestamp"],
                seq["log_uuid"],
            )
        )

    # length is the length of each log sequence (window type) at position 1 (list of events), where each log is an array of [sessionId, eventId, label].
    logs_len = [len(log[2]) for log in logs]
    return logs, {"min": min(logs_len), "max": max(logs_len), "mean": np.mean(logs_len)}


def sliding_window_toindex(
    data: List[Tuple[int, List[int], List[str], List[str], List[str]]],
    history_size: int = 50,
    is_train: bool = True,
    parameter_model: bool = False,
    vocab: Vocab = None,
    sequential: bool = False,
    quantitative: bool = False,
    semantic: bool = False,
    logger: Logger = getLogger("__model__"),
) -> Tuple[
    List[List[int]],
    List[List[int]],
    List[List[int]],
    List[int],
    List[int],
    List[int],
    List[List[int]],
]:
    """
    Description:
        Convert log sequences to indices from vocabulary. Sequentials, quantitatives, semantics, labels, sequentials_idxs, session_ids, parameters

    Parameters:
        data: : List[Tuple[int, List[int], List[str], List[str], List[str]]]: List of log sequences.
        history_size: int: Size of sliding window.
        is_train: bool: training mode or not.
        vocab: Optional[Any]: Vocabulary.
        sequential: bool: Whether to use sequential features.
        quantitative: bool: Whether to use quantitative features.
        semantic: bool: Whether to use semantic features.
        logger: Optional[Any]: Logger.

    Returns:
        sequentials: List[List[int]]: List of sequential features.
        quantitatives: List[List[int]]: List of quantitative features.
        semantics: List[List[int]]: List of semantic features.
        labels: List[int]: List of labels.
        sequentials_idxs: List[int]: List of sequential indices.
        session_ids: List[int]: List of session ids.
        parameters: List[List[int]]: List of parameters.
    """

    log_sequences = []
    sessionIds = {}
    for idx, (sessionId, eventIds, severities, timestamps, log_uuids) in tqdm(
        enumerate(data),
        total=len(data),
        desc=f"Sliding window with size {history_size}",
    ):
        # add pading if the length of the eventIds is less than the history size sequence (when length of eventIds is less than history_size to fullfill the window size e.g current sequence [1,2,3,4,5,6,7,8] but h required is 10, then add 2 more padding to the sequence -> [1,2,3,4,5,6,7,8,token,token] all this within the window size [[1,2,3,4,5,6,7,8,9,10], ... ,[1,2,3,4,5,6,7,8,token,token]]
        eventIds = eventIds + [vocab.pad_token] * (history_size - len(eventIds) + 1)

        sessionIds[idx] = sessionId
        for i in range(len(eventIds) - history_size):
            # get the index of the event from the vocab
            label = vocab.get_event(
                eventIds[i + history_size], use_similar=quantitative
            )  # use_similar only for LogAnomaly

            seq = eventIds[i : i + history_size]
            sequential_pattern = [
                vocab.get_event(event, use_similar=quantitative) for event in seq
            ]

            sequence = {"step": i, "sequential": sequential_pattern}
            sequence["label"] = label
            sequence["idx"] = idx
            if quantitative:
                quantitative_pattern = [0] * len(vocab)
                log_counter = Counter(sequential_pattern)
                for key in log_counter:
                    try:
                        quantitative_pattern[key] = log_counter[key]
                    except Exception as _:
                        pass  # ignore unseen events or padding key
                sequence["quantitative"] = quantitative_pattern

            # build parameters value matrix
            # if parameter_model:
            #     current_timestamp = int(timestamps[i])
            #     last_timestamp = 0
            #     if i != 0:
            #         last_timestamp = int(timestamps[i-1])
            #     t_difference = current_timestamp - last_timestamp
            #     sequence["parameters_values"] = np.array(
            #         [t_difference, severities[i]])
            log_sequences.append(sequence)

    sequentials, quantitatives, semantics = None, None, None
    if sequential:
        sequentials = [seq["sequential"] for seq in log_sequences]
        steps = [seq["step"] for seq in log_sequences]
    if quantitative:
        quantitatives = [seq["quantitative"] for seq in log_sequences]
    if semantic:
        semantics = [seq["semantic"] for seq in log_sequences]
    # if parameter_model:
    #     parameters = [seq['parameters_values']
    #                   for seq in log_sequences]
    labels = [seq["label"] for seq in log_sequences]
    sequentials_idxs = [seq["idx"] for seq in log_sequences]
    logger.info(f"Number of sequences: {len(labels)}")

    return (
        sequentials,
        quantitatives,
        semantics,
        labels,
        sequentials_idxs,
        sessionIds,
        [],
        steps,
    )
