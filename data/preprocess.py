import torch
from data.vocab import Vocab
from typing import Tuple
from data.feature_extraction import load_features, sliding_window
from data.store import Store
from sklearn.utils import shuffle
from data.dataset import LogDataset
import numpy as np


def preprocess_data(path: str,
                    args,
                    is_train: bool,
                    store: Store,
                    logger) -> Tuple[list, list] or list:
    data, stat = load_features(path, is_train=is_train, store=store)
    phase_message = "Train" if is_train else "Test"
    logger.info(
        f"{phase_message} data length: {len(data)}, statistics: {stat}")
    if is_train:
        n_valid = int(len(data) * args.valid_ratio)
        train_data, valid_data = data[:-n_valid], data[-n_valid:]

        store.set_train_data(train_data)
        store.set_valid_data(valid_data)
        store.set_lengths(train_length=len(train_data), valid_length=len(
            valid_data))
        return train_data, valid_data

    else:
        test_data = data
        store.set_test_data(test_data)
        store.set_lengths(test_length=len(test_data))
        label_dict = {}
        counter = {}
        for (k, e, l) in test_data:
            key = tuple(e)
            label_dict[key] = [k, l]
            try:
                counter[key] += 1
            except Exception:
                counter[key] = 1
        test_data = [(list(k), v) for k, v in label_dict.items()]
        num_sessions = [counter[tuple(k)] for k, _ in test_data]
        return test_data, num_sessions


def preprocess_slidings(train_data=None, valid_data=None, test_data=None,
                        vocab=Vocab, args=None,
                        is_train=bool,
                        store=Store,
                        logger=None):

    if is_train:
        sequentials, quantitatives, semantics, labels, sequence_idxs, train_sessionIds = sliding_window(
            train_data,
            vocab=vocab,
            window_size=args.history_size,
            is_train=True,
            semantic=args.semantic,
            quantitative=args.quantitative,
            sequential=args.sequential,
            logger=logger,
        )
        store.set_train_sliding_window(sequentials, quantitatives,
                                       semantics, labels, sequence_idxs, train_sessionIds)

        train_dataset = LogDataset(
            sequentials, quantitatives, semantics, labels, sequence_idxs)

        sequentials, quantitatives, semantics, labels, sequence_idxs, valid_sessionIds = sliding_window(
            valid_data,
            vocab=vocab,
            window_size=args.history_size,
            is_train=True,
            semantic=args.semantic,
            quantitative=args.quantitative,
            sequential=args.sequential,
            logger=logger
        )
        store.set_valid_sliding_window(sequentials, quantitatives,
                                       semantics, labels, sequence_idxs, valid_sessionIds)
        valid_dataset = LogDataset(
            sequentials, quantitatives, semantics, labels, sequence_idxs)
        return train_dataset, valid_dataset, train_sessionIds, valid_sessionIds
    else:
        sequentials, quantitatives, semantics, labels, sequence_idxs, test_sessionIds = sliding_window(
            test_data,
            vocab=vocab,
            window_size=args.history_size,
            is_train=False,
            semantic=args.semantic,
            quantitative=args.quantitative,
            sequential=args.sequential,
            logger=logger
        )
        test_dataset = LogDataset(
            sequentials, quantitatives, semantics, labels, sequence_idxs)
        store.set_test_sliding_window(sequentials, quantitatives,
                                      semantics, labels, sequence_idxs, test_sessionIds)
        return test_dataset, test_sessionIds
