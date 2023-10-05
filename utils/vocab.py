from data.vocab import Vocab
from logging import Logger, getLogger
import pickle
import os


def build_vocab(vocab_path: str,
                data_dir: str,
                train_path: str,
                embeddings: str,
                embedding_dim: int = 300,
                logger: Logger = getLogger("__name__")) -> Vocab:
    """
    Build vocab from training data
    Parameters
    ----------
    vocab_path: str: Path to save vocab
    data_dir: str: Path to data directory
    train_path: str: Path to training data
    embeddings: str: Path to pretrained embeddings
    embedding_dim: int: Dimension of embeddings
    logger: Logger: Logger

    Returns
    -------
    vocab: Vocab: Vocabulary
    """
    if not os.path.exists(vocab_path):
        logger.info(f"Building vocab from {train_path}")
        with open(train_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Lenght of logs {len(data)}")
        logs = [x["EventId"] for x in data]
        vocab = Vocab(logs, os.path.join(data_dir, embeddings),
                      embedding_dim=embedding_dim)
        vocab.save_vocab(vocab_path)
        logger.info(f"Saving vocab in {vocab_path}\n")
    else:
        vocab = Vocab.load_vocab(vocab_path)
        logger.info(f"Loading vocab from {vocab_path}")
    logger.info(f"Vocab size: {len(vocab)}\n")
    return vocab
