from CONSTANTS import *
import yaml
import argparse
import logging
from accelerate import Accelerator
from data.store import Store
from utils.helpers import arg_parser
from utils.utils import build_vocab
from utils.model import build_model
from train import Trainer
from predict import Predicter
from preprocessing.preprocess import Preprocessor
from data.vocab import Vocab

_logger = logging.getLogger("Main")
_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
    )
)

file_handler = logging.FileHandler(os.path.join(LOG_ROOT, "Main.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
    )
)

_logger.addHandler(console_handler)
_logger.addHandler(file_handler)
_logger.info(
    "Construct Main logger success, current working directory: %s, logs will be written in %s"
    % (os.getcwd(), LOG_ROOT)
)
accelerator = Accelerator()

if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()

    if args.config_file is not None and os.path.isfile(args.config_file):
        config_file = args.config_file
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config_args = argparse.Namespace(**config)
            for k, v in config_args.__dict__.items():
                if v is not None:
                    setattr(args, k, v)
        print("\nLoaded config from %s\n" % config_file)
    else:
        print(f"\nLoaded config from command line\n")

    _logger.info(accelerator.state)

    OUTPUT_DIRECTORY = f"{args.output_dir}{args.dataset_folder}/{args.model_name}/train{args.train_size}/h_size{args.window_size}_s_size{args.history_size}/"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIRECTORY}/vocabs", exist_ok=True)

    store = Store(OUTPUT_DIRECTORY, _logger)

    _logger.info("Output directory: %s" % OUTPUT_DIRECTORY)

    preprocessor = Preprocessor(args)
    train_path, test_path = preprocessor.process_dataset(OUTPUT_DIRECTORY)

    setattr(args, "output_dir", OUTPUT_DIRECTORY)
    setattr(args, "device", accelerator.device)
    setattr(args, "accelerator", accelerator)
    setattr(args, "save_dir", f"{OUTPUT_DIRECTORY}/models")
    setattr(args, "train_path", train_path)
    setattr(args, "test_path", test_path)

    vocab_path = f"{OUTPUT_DIRECTORY}vocabs/{args.model_name}.pkl"

    vocab = Vocab()
    exists = vocab.check_already_exists(vocab_path)
    if exists:
        vocab = Vocab.load_vocab(vocab_path)
    else:
        vocab.build_vocab(
            vocab_path,
            train_path,
            embeddings_path=os.path.join(args.data_dir, args.embeddings),
            embedding_dim=args.embedding_dim,
            model_name=args.model_name,
        )
    _logger.info("length of vocabs: %d" % len(vocab))

    model = build_model(args, vocab_size=len(vocab))

    if args.is_train and not args.is_predict:
        trainer = Trainer(model, args, vocab, store)
        trainer.start_training()

    elif args.is_predict and not args.is_train:
        predicter = Predicter(model, vocab, args, store)
        predicter.start_predicting()
    else:
        raise ValueError("Either train, load or update must be True")
