import pickle
import os
import argparse
from time import time
from data.vocab import Vocab
from logging import Logger, getLogger


def build_vocab(
    vocab_path: str,
    train_path: str,
    args: argparse.Namespace,
    logger: Logger = getLogger("__model__"),
) -> Vocab:
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

    if os.path.exists(vocab_path):
        vocab = Vocab.load_vocab(vocab_path)
        logger.info(f"Loading vocab from {vocab_path}")
    else:
        embedding_dim = args.embedding_dim
        data_dir = args.data_dir
        embeddings = args.embeddings
        embeddings_path = os.path.join(data_dir, embeddings)

        logger.info(f"Building vocab from {train_path}")
        with open(train_path, "rb") as f:
            data = pickle.load(f)
        logs = [x["EventId"] for x in data]
        vocab = None

        if args.model_name == "DeepLog":
            print("intiia")
            vocab = Vocab(logs, embeddings_path,
                          embedding_dim, use_similar=False)
        elif args.model_name == "LogAnomaly":
            vocab = Vocab(logs, embeddings_path,
                          embedding_dim, use_similar=True)
        else:
            raise NotImplementedError(
                f"{args.model_name} is not implemented")
        vocab.save_vocab(vocab_path)
        logger.info(f"Saving vocab in {vocab_path}\n")

    logger.info(f"Vocab size: {len(vocab)}\n")
    return vocab


# run as python -m utils.vocab to print saved vocab stoi
if __name__ == "__main__":
    vocab_path = f"output/DeepLog/train0.1/w_size50_s_size50/vocabs/Deeplog.pkl"
    data_dir = f"./dataset"
    train_path = f"output/DeepLog/train0.1/w_size50_s_size50/train.pkl"
    embeddings = f"sep20-21/embeddings_average.json"
    embedding_dim = 300
    logger = None
    vocabs = build_vocab(
        vocab_path, data_dir, train_path, embeddings, embedding_dim, logger
    )
    print(vocabs.stoi)
