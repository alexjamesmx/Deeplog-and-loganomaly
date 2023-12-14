from CONSTANTS import *
import pandas as pd
import json


class Preprocessor:
    _logger = logging.getLogger("Preprocessor")
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
        )
    )

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, "Preprocessor.log"))
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
        return Preprocessor._logger

    def __init__(self, args: argparse.Namespace = None):
        self.df = None
        self.grouping = args.grouping
        self.window_size = args.window_size
        self.step_size = args.step_size
        self.train_size = args.train_size
        self.dataset_folder = args.dataset_folder
        self.log_file = args.log_file
        self.data_dir = args.data_dir
        self.output_directory = ""

        pass

    def df_2_windowedSessions(
        self,
    ) -> Tuple[str, str]:
        """
        Description:
            Creates windowed sessions from df.
        Raises:
            NotImplementedError: _description_

        Returns:
            Tuple[str, str]: train.pkl and test.pkl paths.
        """
        if self.grouping == "sliding":
            window_df = self.create_sliding_windows()

            n_train = int(self.train_size * len(window_df))
            train_window = window_df[:n_train]
            test_window = window_df[n_train:]
            self.logger.info(
                "Total sessions: %d, train sessions: %d, test sessions: %d"
                % (len(window_df), len(train_window), len(test_window))
            )
        else:
            raise NotImplementedError(
                "%s grouping method is not implemented" % self.grouping
            )

        with open(os.path.join(self.output_directory, "train.pkl"), mode="wb") as f:
            pickle.dump(train_window, f)
        with open(os.path.join(self.output_directory, "test.pkl"), mode="wb") as f:
            pickle.dump(test_window, f)
        self.logger.info(f"Saved train.pkl and test.pkl at {self.output_directory}")
        return os.path.join(self.output_directory, "train.pkl"), os.path.join(
            self.output_directory, "test.pkl"
        )

    def process_dataset(self, output_directory: str) -> Tuple[str, str]:
        """
        Load train/text df (windowed sessions).

        Parameters:
        output_directory (str): directory where .pkls are saved.

        Raises:
        NotImplementedError: Data file does not exist.
        NotImplementedError: Grouping method is not implemented (for now only sliding is implemented).

        Returns:
        Tuple[str, str]: train.pkl and test.pkl paths.
        """

        self.output_directory = output_directory
        train_path = os.path.join(self.output_directory, "train.pkl")
        test_path = os.path.join(self.output_directory, "test.pkl")

        if os.path.exists(train_path) and os.path.exists(test_path):
            self.logger.info(
                "Loading train.pkl and test.pkl from %s" % self.output_directory
            )
            return os.path.join(self.output_directory, "train.pkl"), os.path.join(
                self.output_directory, "test.pkl"
            )

        data_path = f"{self.data_dir}{self.dataset_folder}{self.log_file}"

        self.df = self.load_data_from_files(data_path)
        return self.df_2_windowedSessions()

    def create_sliding_windows(self) -> List[Dict[str, List]]:
        """
        Description:
            Generate fixed/sliding window for the dataset.
            if step_size == window_size, then it is fixed window.
            if step_size < window_size, then it is sliding window.

        Returns:
            new_data (List[Dict[str, List]]): List of log sequences
        """
        log_size = self.df.shape[0]
        (
            timestamp,
            severity,
            events,
            message,
            log_uuid,
        ) = (
            self.df["_zl_timestamp"],
            self.df["SEVERITY"],
            self.df["EVENTID"],
            self.df["MESSAGE"],
            self.df["log_uuid"],
        )

        # severity_values = df["SEVERITY"].unique()
        # print(f"Severity values: {severity_values}")
        # severity_mapping = {severity: i for i,
        # severity in enumerate(severity_values)}
        # print(f"Severity mapping: {severity_mapping}")
        # Apply the mapping to create a new numerical column
        # severity = df['SEVERITY'].map(severity_mapping)

        new_data = []
        start_end_index_pair = []

        start_index = 0
        while start_index < log_size:
            end_index = min(start_index + self.window_size, log_size)
            start_end_index_pair.append(tuple([start_index, end_index]))
            start_index = start_index + self.step_size

        n_sess = 0
        for start_index, end_index in start_end_index_pair:
            new_data.append(
                {
                    "SessionId": n_sess,
                    "_zl_timestamp": timestamp[start_index:end_index].values.tolist(),
                    "SEVERITY": severity[start_index:end_index].values.tolist(),
                    "EventId": events[start_index:end_index].values.tolist(),
                    "MESSAGE": message[start_index:end_index].values.tolist(),
                    "log_uuid": log_uuid[start_index:end_index].values.tolist(),
                }
            )
            n_sess += 1

        assert len(start_end_index_pair) == len(new_data)
        return new_data

    def load_data_from_files(self, data_path: str) -> pd.DataFrame:
        """
        Load or create json file from txt file

        Raises:
            NotImplementedError: If txt file path does not exist
        Returns:
            df: dataframe of json file
        """
        json_path = f"{data_path}.json"
        txt_path = f"{data_path}.txt"

        if os.path.isfile(json_path):
            df = pd.read_json(json_path, orient="records", lines=True)
        else:
            if not os.path.isfile(txt_path):
                raise NotImplementedError(f"The file {txt_path}.txt does not exist")
            self.logger.info("Creating json file from txt file: %s" % txt_path)
            logs, count_errors = self.raw_logs_tojson(txt_path, num_lines=100000)
            self.count_keys(logs)  # Comment this line if you dont want to count keys
            self.logger.info(
                f"Total logs in {txt_path}: {len(logs)}, errors {count_errors}"
            )
            df = pd.DataFrame(logs)
            self.logger.info("Saving json in %s" % json_path)
            df.to_json(json_path, orient="records", lines=True)
        return df

    def raw_logs_tojson(self, data_dir, num_lines=None) -> Tuple[List[dict], int]:
        """
        Description:
            Parses txt log rows to json objects.
            Discards rows that are not in json format.

        Parameters:
            data_dir (str): path to txt file
            num_lines (int, optional): number of lines to read.

        Returns:
            Tuple[List[dict], int]: List of json objects and number of parsing errors
        """
        json_objects = []
        count_errors = 0
        line_count = 0
        no_eventId = 0
        with open(data_dir, "r") as f:
            for line in f:
                if num_lines is not None and line_count >= num_lines:
                    break
                try:
                    json_obj = json.loads(line)
                    if isinstance(json_obj, dict):
                        if json_obj.get("EVENTID") is None:
                            # skip if no eventId
                            no_eventId += 1
                        else:
                            json_objects.append(json_obj)
                            line_count += 1
                    else:
                        self.logger.info("Error: not a dict: %s" % line)
                        count_errors += 1
                except json.JSONDecodeError as e:
                    print(e)
                    count_errors += 1
            self.logger.info("No Total eventIds: %d" % no_eventId)
        return json_objects, count_errors

    def count_keys(self, logs: List[dict]):
        """(debugging)
        Description:
            Counts keys from dataset.

        Note:
            Not all logs have the same keys (unknown keys might appear).
            Deeplog/LogAnomaly use eventTemplates, however, our data does not have this key so we use evenIDs
            Not every logs have a eventId (causing trouble).

        Parameters:
            logs (List[dict]):

        """
        counter = Counter()
        for log in logs:
            counter.update(log.keys())
        self.logger.info("Total keys: %d" % counter)
