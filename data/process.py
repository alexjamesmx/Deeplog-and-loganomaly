import argparse
from logging import Logger, getLogger
from data.vocab import Vocab
from typing import Tuple, List
from data.feature_extraction import load_features, sliding_window_toindex
from data.store import Store
from data.dataset import LogDataset


def process_sessions(
    path: str,
    args: argparse.Namespace,
    is_train: bool,
    store: Store = None,
    logger: Logger = getLogger("__model__"),
) -> Tuple[int, List[str], List[str], List[str], List[str]]:
    """
    Description:
        Split sessions to train, valid and test.
        Store train, valid and test data in store class.

    Parameters:
        path (str): path to train or test data.
        args (argparse.Namespace).
        is_train (bool).
        store (Store).
        logger (Logger).

    Returns:
        Tuple[int, List[str], List[str], List[str], List[str]].
        e.g:
            (session_id, list event_ids, list_severities, list timestamps, list log_uuids).
    """

    data, stat = load_features(path, is_train=is_train, store=store)

    phase_message = "Train" if is_train else "Test"
    logger.info(
        f"{phase_message} data length: {len(data)} (sessions), statistics: {stat} total_train_logs {len(data * args.window_size)}"
    )
    if is_train:
        data = data
        n_valid = int(len(data) * args.valid_ratio)
        train_data, valid_data = data[:-n_valid], data[-n_valid:]
        # testing
        # data = data[:4]
        # train_data, valid_data = data[:3], data[-1:]
        store.set_train_data(train_data)
        store.set_valid_data(valid_data)
        store.set_lengths(train_length=len(train_data), valid_length=len(valid_data))
        logger.info(
            f"Size of train data: {len(train_data)}. Size of valid data: {len(valid_data)}"
        )
        return train_data, valid_data

    else:
        # testing
        # test_data = data[:4000]
        test_data = data
        store.set_lengths(test_length=len(test_data))
        # print(test_data)

        # num_sessions = [1 for _ in test_data]
        logger.info(
            f"No. sessions test data: {len(test_data)} | window size: {args.window_size} | history size: {args.history_size} | step: {1} | total_sequences {args.history_size * len(test_data)}"
        )
        return test_data


def create_datasets(
    train_data: List[Tuple[int, List[int], List[str], List[str], List[str]]] = None,
    valid_data: List[Tuple[int, List[int], List[str], List[str], List[str]]] = None,
    test_data: List[Tuple[int, List[int], List[str], List[str], List[str]]] = None,
    vocab: Vocab = None,
    args: argparse.Namespace = None,
    is_train: bool = True,
    store: Store = None,
    logger: Logger = getLogger("__model__"),
) -> Tuple[LogDataset, LogDataset, list, list] or Tuple[LogDataset, list]:
    """
    Description:
        Transform sessions to sliding windows.
        Store train, valid and test sliding windows in store class and return datasets.
        E.g see at ./testing/slidings.txt.
    Args:
        train_data (List[Tuple[int, List[int], List[str], List[str], List[str]]]): train sessions.
        valid_data (List[Tuple[int, List[int], List[str], List[str], List[str]]]): valid sessions.
        test_data (List[Tuple[int, List[int], List[str], List[str], List[str]]]): Test sessions.
        vocab (Vocab).
        args (argparse.Namespace).
        is_train (bool).
        store (Store).
        logger (Logger).
        one sample:
            (2200, [4658, 4658, 4690, 4658, 4656, 4663, 4690, 4658, 4656, 4658], ['success', 'success', 'success', 'success', 'success', 'success', 'success', 'success', 'success', 'success'], [1695204173000, 1695204173000, 1695204173000, 1695204173000, 1695204173000, 1695204173000, 1695204173000, 1695204173000, 1695204173000, 1695204173000], ['140ff65f-579d-11ee-988e-cc483a4638dd_elacollector8da9fe5e9b2c42c7843bcdff5175e0435257320593532651', '140ff65e-579d-11ee-988e-cc483a4638dd_elacollector8da9fe5e9b2c42c7843bcdff5175e0435257320593532651', '140ff65d-579d-11ee-988e-cc483a4638dd_elacollector8da9fe5e9b2c42c7843bcdff5175e0435257320593532651', '140ff65c-579d-11ee-988e-cc483a4638dd_elacollector8da9fe5e9b2c42c7843bcdff5175e0435257320593532651', '140ff65b-579d-11ee-988e-cc483a4638dd_elacollector8da9fe5e9b2c42c7843bcdff5175e0435257320593532651', '140ff65a-579d-11ee-988e-cc483a4638dd_elacollector8da9fe5e9b2c42c7843bcdff5175e0435257320593532651', '140ff659-579d-11ee-988e-cc483a4638dd_elacollector8da9fe5e9b2c42c7843bcdff5175e0435257320593532651', '140ff658-579d-11ee-988e-cc483a4638dd_elacollector8da9fe5e9b2c42c7843bcdff5175e0435257320593532651', '140ff657-579d-11ee-988e-cc483a4638dd_elacollector8da9fe5e9b2c42c7843bcdff5175e0435257320593532651', '140ff656-579d-11ee-988e-cc483a4638dd_elacollector8da9fe5e9b2c42c7843bcdff5175e0435257320593532651']))
            (session_id, list event ids, list severities, list timestamps, list log_uuids)
    Returns:
        Tuple[LogDataset, LogDataset, list, list] or Tuple[LogDataset, list].
    """
    if is_train:
        (
            sequentials,
            quantitatives,
            semantics,
            labels,
            sequentials_idxs,
            _,
            train_parameters,
            _,
        ) = sliding_window_toindex(
            train_data,
            vocab=vocab,
            history_size=args.history_size,
            is_train=True,
            parameter_model=args.parameter_model,
            semantic=args.semantic,
            quantitative=args.quantitative,
            sequential=args.sequential,
            logger=logger,
        )
        store.set_train_sliding_window(
            sequentials, quantitatives, semantics, labels, sequentials_idxs
        )

        train_dataset = LogDataset(
            sequentials, quantitatives, semantics, labels, sequentials_idxs
        )

        (
            sequentials,
            quantitatives,
            semantics,
            labels,
            sequentials_idxs,
            valid_sessionIds,
            valid_parameters,
            _,
        ) = sliding_window_toindex(
            valid_data,
            vocab=vocab,
            history_size=args.history_size,
            is_train=True,
            parameter_model=args.parameter_model,
            semantic=args.semantic,
            quantitative=args.quantitative,
            sequential=args.sequential,
            logger=logger,
        )
        store.set_valid_sliding_window(
            sequentials,
            quantitatives,
            semantics,
            labels,
            sequentials_idxs,
            valid_sessionIds,
        )
        valid_dataset = LogDataset(
            sequentials,
            quantitatives,
            semantics,
            labels,
            sequentials_idxs,
            valid_sessionIds,
        )
        return train_dataset, valid_dataset, train_parameters, valid_parameters

    else:
        (
            sequentials,
            quantitatives,
            semantics,
            labels,
            sequentials_idxs,
            test_sessionIds,
            test_parameters,
            steps,
        ) = sliding_window_toindex(
            test_data,
            vocab=vocab,
            history_size=args.history_size,
            is_train=True,
            parameter_model=args.parameter_model,
            semantic=args.semantic,
            quantitative=args.quantitative,
            sequential=args.sequential,
            logger=logger,
        )
        test_dataset = LogDataset(
            sequentials,
            quantitatives,
            semantics,
            labels,
            sequentials_idxs,
            test_sessionIds,
            steps,
        )
        store.set_test_sliding_window(
            sequentials,
            quantitatives,
            semantics,
            labels,
            sequentials_idxs,
            test_sessionIds,
            steps,
        )
        return test_dataset, test_parameters
