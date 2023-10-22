import os
import yaml
import argparse
import logging
from torch import nn

from logging import getLogger, Logger
from accelerate import Accelerator

from data.data_loader import process_dataset
from data.store import Store

from utils.helpers import arg_parser
from utils.vocab import build_vocab
from utils.model import build_model

from train import Trainer
from predict import Predicter

from data.vocab import Vocab


def run_train(args: argparse.Namespace,
              model: nn.Module,
              vocabs: Vocab,
              store: Store):
    """
    Trains model
    Args:
        args (argparse.Namespace): Arguments 
        store (Store)
    """
    args.logger.info("Start training")
    trainer = Trainer(model,
                      args,
                      vocabs,
                      store)
    trainer.start_training()


def run_predict(args: argparse.Namespace,
                model: nn.Module,
                vocabs: Vocab,
                store: Store):
    """
    Predicts a dataset
    Args:
        args (argparse.Namespace): Arguments 
        store (Store)
    """
    logger.info("Start predicting")
    predicter = Predicter(model,
                          vocabs,
                          args,
                          store)
    predicter.start_predicting()


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

accelerator = Accelerator()

if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()

    logger = getLogger(args.model_name)
    logger.info(accelerator.state)
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    if args.config_file is not None and os.path.isfile(args.config_file):
        config_file = args.config_file
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config_args = argparse.Namespace(**config)
            for k, v in config_args.__dict__.items():
                if v is not None:
                    setattr(args, k, v)
        logger.info(f"Loaded config from {config_file}")
    else:
        logger.info(f"Loaded config from command line")

    output_dir = f"{args.output_dir}{args.dataset_folder}/{args.model_name}/train{args.train_size}/h_size{args.window_size}_s_size{args.history_size}/"
    os.makedirs(f"{output_dir}/vocabs", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    store = Store(output_dir, logger)

    logger.info(f"Output directory: {output_dir}")
    # obtain .pkl's  paths
    train_path, test_path = process_dataset(args,
                                            output_dir=output_dir,
                                            logger=logger)
    # set attributes
    setattr(args, "output_dir", output_dir)
    setattr(args, "device", accelerator.device)
    setattr(args, "logger", logger)
    setattr(args, "accelerator", accelerator)
    setattr(args, "save_dir", f"{output_dir}/models")
    setattr(args, "train_path", train_path)
    setattr(args, "test_path", test_path)

    # load vocab
    vocab_path = f"{output_dir}vocabs/{args.model_name}.pkl"

    vocabs = build_vocab(vocab_path,
                         data_dir=args.data_dir,
                         train_path=train_path,
                         embeddings=args.embeddings,
                         embedding_dim=args.embedding_dim,
                         logger=logger)
    model = build_model(args, vocab_size=len(vocabs))

    if args.is_train and not args.is_predict:
        run_train(args,
                  model,
                  vocabs,
                  store)
    elif args.is_predict and not args.is_train:
        run_predict(args,
                    model,
                    vocabs,
                    store)
    else:
        raise ValueError("Either train, load or update must be True")
