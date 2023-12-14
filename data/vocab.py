from collections import Counter
import pickle
import json
import os
from CONSTANTS import *

import numpy as np
from numpy import dot
from numpy.linalg import norm
import math


def read_json(filename):
    with open(filename, "r") as load_f:
        file_dict = json.load(load_f)
    return file_dict


class Vocab(object):
    _logger = logging.getLogger("Vocab")
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
        )
    )

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, "Vocab.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
        )
    )

    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        "Construct Vocab logger success, current working directory: %s, logs will be written in %s"
        % (os.getcwd(), LOG_ROOT)
    )

    @property
    def logger(self):
        return Vocab._logger

    def __init__(
        self,
    ):
        self.embeddings_path = None
        self.embedding_dim = None
        self.use_similar = None
        self.emb_file = None

        self.logs = []
        self.stoi = {}
        self.itos = ["padding"]
        self.pad_token = "padding"

        self.unk_index = None
        # NOTE add indices where e is the event and i is the index
        self.stoi = {}
        self.semantic_vectors = {}

        self.mapping = {}

    def __len__(self):
        return len(self.itos)

    def get_event(self, real_event, use_similar=False) -> int:
        """Get event index from vocab
        Args:
            real_event (str): real log event
            use_similar (bool): whether to use similar events
        Returns:
            int: event index
        """
        event = self.stoi.get(real_event, self.unk_index)

        if not use_similar or event != self.unk_index:
            return event

        if self.mapping.get(real_event) is not None:
            return self.mapping[real_event]

        for train_event in self.itos[:-1]:
            real_event = str(real_event)
            sim = dot(
                self.semantic_vectors[real_event],
                self.semantic_vectors[str(train_event)],
            ) / (
                norm(self.semantic_vectors[real_event])
                * norm(self.semantic_vectors[str(train_event)])
            )
            if sim > 0.90:
                self.mapping[real_event] = self.stoi.get(train_event)
                # print(
                #     f"event {real_event} mapped to {train_event} -- {self.stoi.get(train_event)}")
                return self.stoi.get(train_event)
        self.mapping[real_event] = self.unk_index
        return self.mapping[real_event]

    def get_embedding(self, event) -> list:
        return self.semantic_vectors[event]

    def save_vocab(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    def build_vocab(
        self,
        vocab_path: str,
        train_path: str,
        embeddings_path: str,
        embedding_dim: int,
        model_name: str,
    ):
        """
        Build vocab from training data.

        Parameters:
            vocab_path: str: Path to save vocab
            train_path: str: Path to training data
            args (argparse.Namespace): Arguments
            logger (Logger): Logger

        Raises:
            NotImplementedError: Model name is not implemented

        Returns:
            vocab (Vocab): Vocabulary
        """
        self.embeddings_path = embeddings_path
        self.embedding_dim = embedding_dim

        self.logger.info(f"Building vocab from {train_path}")
        with open(train_path, "rb") as f:
            data = pickle.load(f)
        self.logs = [x["EventId"] for x in data]

        if model_name == "DeepLog":
            print("intiia")
            self.use_similar = False
        elif model_name == "LogAnomaly":
            self.use_similar = True
        else:
            raise NotImplementedError(f"{model_name} is not implemented")

        self.update_vocab()

        self.save_vocab(vocab_path)
        self.logger.info(f"Saving vocab in {vocab_path}\n")

        self.logger.info(f"Vocab size: {len(self)}\n")

    def update_vocab(self):
        for line in self.logs:
            self.itos.extend(line)
        self.itos = ["padding"] + list(set(self.itos))

        self.unk_index = len(self.itos)
        # NOTE add indices where e is the event and i is the index
        self.stoi = {e: i for i, e in enumerate(self.itos)}
        if self.use_similar:
            self.semantic_vectors = read_json(self.emb_file)

            self.semantic_vectors = {
                k: v if type(v) is list else [0] * self.embedding_dim
                for k, v in self.semantic_vectors.items()
            }
            # NOTE add token at the end of the vocab
            self.semantic_vectors[self.pad_token] = [-1] * self.embedding_dim

    def check_already_exists(self, vocab_path):
        return os.path.exists(vocab_path)

    @staticmethod
    def load_vocab(file_path):
        with open(file_path, "rb") as f:
            loaded_vocab = pickle.load(f)
        return loaded_vocab
