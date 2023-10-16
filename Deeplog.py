import os
import yaml
import torch
import argparse
import logging

from logging import getLogger, Logger
from accelerate import Accelerator
from typing import Tuple

from data.vocab import Vocab
from data.data_loader import process_dataset
from data.preprocess import preprocess_data, preprocess_slidings
from data.store import Store

from trainer import Trainer

from utils.helpers import arg_parser, get_optimizer
from utils.vocab import build_vocab
from utils.model import build_model

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

accelerator = Accelerator()


def run_train(args: argparse.Namespace,
              output_dir: str,
              store: Store,
              logger: Logger = getLogger("__name__")):
    """
    Preprocess and train model

    Args:
        args (argparse.Namespace): Arguments 
        output_dir (str): Output directory
        store (Store): Class to store original and preprocessed data
        logger (Logger)

    """
    logger.info("Start training")
    train_path, test_path = process_dataset(logger=logger,
                                            output_dir=output_dir,
                                            args=args)
    vocab_path = f"{output_dir}vocabs/{args.model_name}.pkl"
    vocabs = build_vocab(vocab_path,
                         args.data_dir,
                         train_path,
                         args.embeddings,
                         args.embedding_dim,
                         logger)
    model = build_model(args, vocab_size=len(vocabs))

    train(args,
          train_path,
          test_path,
          vocabs,
          model,
          store,
          output_dir,
          logger,
          accelerator)


def train(args: argparse.Namespace,
          train_path: str,
          test_path: str,
          vocab: Vocab,
          model: torch.nn.Module,
          store: Store,
          output_dir: str,
          logger: Logger = getLogger("__name__"),
          accelerator: Accelerator = Accelerator()
          ) -> Tuple[float, float]:
    """
    Process data
    Train model

    Args:
        args (argparse.Namespace): Arguments
        train_path (str): Path to training data
        test_path (str): Path to test data
        vocab (Vocab): Vocabulary
        model (torch.nn.Module): Model
        store (Store): log store 
        output_dir (str): Output directory
        logger (Logger, optional) 
        accelerator (Accelerator) 

    Returns:
        Tuple[float, float]: Metrics
    """
    # get train and valid data
    train_data, valid_data = preprocess_data(
        path=train_path,
        args=args,
        is_train=True,
        store=store,
        logger=logger)
    # turn event_ids sequences into vocab indexes and pad them
    # parameters still not implemented
    train_dataset, valid_dataset, train_parameters, valid_parameters = preprocess_slidings(
        train_data=train_data,
        valid_data=valid_data,
        vocab=vocab,
        args=args,
        is_train=True,
        store=store,
        logger=logger,
    )
    # valid session indexes for recommending topk only (evaluation)
    valid_session_idxs = valid_dataset.get_session_labels()

    # print("parameters ", train_parameters)

    optimizer = get_optimizer(args, model.parameters())

    device = accelerator.device
    model = model.to(device)
    if args.see_config:
        logger.info(f"Optimizer: {optimizer}")
        logger.info(model)

    trainer = Trainer(
        model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        is_train=True,
        optimizer=optimizer,
        no_epochs=args.max_epoch,
        batch_size=args.batch_size,
        scheduler_type=args.scheduler,
        warmup_rate=args.warmup_rate,
        accumulation_step=args.accumulation_step,
        logger=logger,
        accelerator=accelerator,
        num_classes=len(vocab),
    )

    logger.info(
        f"Start training {args.model_name} model on {device} device\n")

    # print("1 train data ", train_dataset[0])

    train_loss, val_loss, val_acc, _ = trainer.train(device=device,
                                                     save_dir=f"{output_dir}/models",
                                                     model_name=args.model_name,
                                                     topk=args.topk)
    logger.info(
        f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
    minimum_recommended_topk = trainer.predict_unsupervised(valid_dataset,
                                                            valid_session_idxs,
                                                            topk=args.topk,
                                                            device=device,
                                                            is_valid=True)
    logger.info(
        f"Top-{args.topk} Min recommendation: {minimum_recommended_topk}\n")

    # Get test data
    test_data = preprocess_data(
        path=test_path,
        args=args,
        is_train=False,
        store=store,
        logger=logger)
    # print("this is how my data looks ", test_data[0])
    # turn event_ids sequences into vocab indexes and pad them
    test_dataset, parameters = preprocess_slidings(
        test_data=test_data,
        vocab=vocab,
        args=args,
        is_train=False,
        store=store,
        logger=logger,
    )
    # obtain session indexes
    session_ids = test_dataset.get_session_labels()
    print(vocab.stoi)

    # store.lengths
    # store.get_test_sliding_window(length=True)
    logger.info(
        f"Start predicting {args.model_name} model on {device} device with top-{args.topk} recommendation")

    normal, anomalies = trainer.predict_unsupervised(dataset=test_dataset,
                                                     #  y_true=[],
                                                     topk=args.topk,
                                                     device=device,
                                                     is_valid=False,
                                                     #  num_sessions=num_sessions,
                                                     session_ids=session_ids,
                                                     args=args,
                                                     store=store,
                                                     )
    logger.info(
        f"Normal: {normal} - Anomalies: {anomalies}")


def run_predict(args: argparse.Namespace,
                output_dir: str,
                store: Store,
                logger: Logger = getLogger("__name__")):
    logger.info("Start predicting :)")
    train_path, test_path = process_dataset(logger=logger,
                                            output_dir=output_dir,
                                            args=args)
    vocab_path = f"{output_dir}vocabs/{args.model_name}.pkl"
    vocabs = build_vocab(vocab_path,
                         args.data_dir,
                         train_path,
                         args.embeddings,
                         args.embedding_dim,
                         logger)
    model = build_model(args, vocab_size=len(vocabs))
    normal, anomalies = predict(args,
                                test_path,
                                vocabs,
                                model,
                                store,
                                output_dir,
                                logger,
                                accelerator)
    logger.info(
        f"Normal: {normal} - Anomalies: {anomalies} ")


def predict(args: argparse.Namespace,
            test_path: str,
            vocab: Vocab,
            model: torch.nn.Module,
            store: Store,
            output_dir: str,
            logger: Logger = getLogger("__name__"),
            accelerator: Accelerator = Accelerator()
            ) -> Tuple[float, float]:
    """
    Predict model

    Args:
        args (argparse.Namespace): Arguments
        test_path (str): Path to test data
        vocab (Vocab): Vocabulary
        model (torch.nn.Module): Model
        store (Store): log store 
        output_dir (str): Output directory
        logger (Logger, optional) 
        accelerator (Accelerator) 

    Returns:
        Tuple[float, float]: Metrics
    """
    optimizer = get_optimizer(args, model.parameters())
    device = accelerator.device
    model = model.to(device)
    if args.see_config:
        logger.info(f"Optimizer: {optimizer}")
        logger.info(model)

    trainer = Trainer(
        model,
        is_train=False,
        optimizer=optimizer,
        no_epochs=args.max_epoch,
        batch_size=args.batch_size,
        scheduler_type=args.scheduler,
        warmup_rate=args.warmup_rate,
        accumulation_step=args.accumulation_step,
        logger=logger,
        accelerator=accelerator,
        num_classes=len(vocab),
    )

    trainer.load_model(f"{output_dir}/models/{args.model_name}.pt")

    test_data, num_sessions = preprocess_data(
        path=test_path,
        args=args,
        is_train=False,
        store=store,
        logger=logger)

    test_dataset,  parameteres = preprocess_slidings(
        test_data=test_data,
        vocab=vocab,
        args=args,
        is_train=False,
        store=store,
        logger=logger,
    )
    session_ids = test_dataset.get_session_labels()
    # print(session_ids)

    # # store.lengths
    # store.get_test_sliding_window()
    print(vocab.stoi)
    logger.info(
        f"Start predicting {args.model_name} model on {device} device with top-{args.topk} recommendation")

    normal, anomalies = trainer.predict_unsupervised(dataset=test_dataset,
                                                     y_true=[],
                                                     topk=args.topk,
                                                     device=device,
                                                     is_valid=False,
                                                     num_sessions=num_sessions,
                                                     session_ids=session_ids,
                                                     args=args,
                                                     store=store,
                                                     )
    logger.info(
        f"Normal: {normal} - Anomalies: {anomalies}")


if __name__ == "__main__":
    # basic config
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
        logger.info(f"Loaded config from {config_file}!")
    else:
        logger.info(f"Loaded config from command line!")

    output_dir = f"{args.output_dir}{args.dataset_folder}/{args.model_name}/train{args.train_size}/h_size{args.window_size}_s_size{args.history_size}/"

    os.makedirs(f"{output_dir}/vocabs", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    store = Store(output_dir, logger)

    logger.info(f"Output directory: {output_dir}")

    if args.is_train and not args.is_load:
        run_train(args, output_dir, store, logger)
    elif args.is_load and not args.is_train:
        run_predict(args, output_dir, store, logger)
    else:
        raise ValueError("Either train, load or update must be True")
