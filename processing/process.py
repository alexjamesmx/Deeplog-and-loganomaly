from CONSTANTS import *
from data.store import Store
from data.vocab import Vocab
from data.dataset import LogDataset


class Processor:
    _logger = logging.getLogger("Processor")
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
        )
    )

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, "Processor.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
        )
    )

    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        "Construct Preprocessor logger success, current working directory: %s, logs will be written in %s"
        % (os.getcwd(), LOG_ROOT)
    )

    @property
    def logger(self):
        return Processor._logger

    def __init__(
        self,
    ):
        pass

    def split_sessions(
        self,
        sessions_path: str,
        valid_ratio: float = 0.2,
        is_train: bool = True,
        store: Store = None,
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
        self.sessions_path = sessions_path
        self.is_train = is_train

        data, stat = load_features(
            data_path=sessions_path,
            is_train=is_train,
            store=store,
        )

        self.logger.info
        if is_train:
            data = data
            n_valid = int(len(data) * valid_ratio)
            train_data, valid_data = data[:-n_valid], data[-n_valid:]
            store.set_train_data(train_data)
            store.set_valid_data(valid_data)
            store.set_lengths(
                train_length=len(train_data), valid_length=len(valid_data)
            )
            self.logger.info(
                "Train sessions size: %d. Valid sessions size: %d. statistics: %s"
                % (len(train_data), len(valid_data), stat)
            )
            return train_data, valid_data

        else:
            test_data = data
            store.set_lengths(test_length=len(test_data))
            self.logger.info(
                "Test sessions size: %d. statistics: %s" % (len(test_data), stat)
            )

            return test_data

    def create_datasets(
        self,
        train_data: List[Tuple[int, List[int], List[str], List[str], List[str]]] = None,
        valid_data: List[Tuple[int, List[int], List[str], List[str], List[str]]] = None,
        test_data: List[Tuple[int, List[int], List[str], List[str], List[str]]] = None,
        vocab: Vocab = None,
        history_size: int = 50,
        parameter_model: bool = False,
        semantic: bool = False,
        quantitative: bool = False,
        sequential: bool = False,
        is_train: bool = True,
        store: Store = None,
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
            ) = slidingWindow_2index(
                train_data,
                vocab=vocab,
                history_size=history_size,
                is_train=True,
                parameter_model=parameter_model,
                semantic=semantic,
                quantitative=quantitative,
                sequential=sequential,
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
            ) = slidingWindow_2index(
                valid_data,
                vocab=vocab,
                history_size=history_size,
                is_train=True,
                parameter_model=parameter_model,
                semantic=semantic,
                quantitative=quantitative,
                sequential=sequential,
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
            ) = slidingWindow_2index(
                test_data,
                vocab=vocab,
                history_size=history_size,
                is_train=True,
                parameter_model=parameter_model,
                semantic=semantic,
                quantitative=quantitative,
                sequential=sequential,
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


def slidingWindow_2index(
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
