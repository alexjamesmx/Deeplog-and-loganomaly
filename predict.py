import torch
import numpy as np
import os
import argparse

from datetime import datetime
from data.dataset import LogDataset

from utils.helpers import get_optimizer
from utils.helpers import evaluate_predictions

from data.store import Store
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
from CONSTANTS import *
from processing.process import Processor


class Predicter:
    _logger = logging.getLogger("Predicter")
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
        )
    )

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, "Predicter.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
        )
    )

    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        "Construct Predicter logger success, current working directory: %s, logs will be written in %s"
        % (os.getcwd(), LOG_ROOT)
    )

    @property
    def logger(self):
        return Predicter._logger

    def __init__(self, model, vocabs, args, store):
        self.accelerator = args.accelerator
        self.device = self.accelerator.device
        self.model = model.to(self.device)
        self.optimizer = get_optimizer(args, self.model.parameters())
        self.is_train = False
        self.no_epochs = args.max_epoch
        self.scheduler_type = args.scheduler
        self.batch_size = args.batch_size
        self.warmup_rate = args.warmup_rate
        self.accumulation_step = args.accumulation_step
        self.num_classes = len(vocabs)
        self.model_name = args.model_name
        self.args = args
        self.path = args.test_path
        self.topk = args.topk
        self.store = store
        self.scheduler = None

        processor = Processor()

        self.test_sessions = processor.split_sessions(
            sessions_path=self.path,
            is_train=self.is_train,
            store=store,
        )

        self.test_dataset, self.test_parameters = processor.create_datasets(
            test_data=self.test_sessions,
            vocab=vocabs,
            history_size=self.args.history_size,
            parameter_model=self.args.parameter_model,
            semantic=self.args.semantic,
            quantitative=self.args.quantitative,
            sequential=self.args.sequential,
            is_train=self.is_train,
            store=store,
        )
        self.session_ids = self.test_dataset.get_session_ids()

    def predict_unsupervised(
        self,
        dataset: LogDataset,
        session_ids: List[int],
        topk: int = 9,
        device: str = "cpu",
        args: argparse.Namespace = None,
        store: Store = None,
    ) -> int:
        """
        Description:
            Predict the top-k labels for each sequence in the dataset.

        Parameters:
            dataset (LogDataset): dataset to predict.
            session_ids (List[int]): List of session ids of dataset to predict.
            topk (int): top-k labels to predict.
            device (str).
            args (argparse.Namespace).
            store (Store): Get original logs.
        Returns:
            int: Number of anomalies.
        """

        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.to(device)
        self.model, test_loader = self.accelerator.prepare(self.model, test_loader)
        self.model.eval()

        return self.predict_unsupervised_helper(
            test_loader, session_ids, topk, args, store, device
        )

    def predict_unsupervised_helper(
        self,
        test_loader,
        session_ids: List,
        topk: int = 9,
        args: argparse.Namespace = None,
        store=Store,
        device: str = "cpu",
    ) -> int:
        """
        Description:
            Counts and creates anomalies file.

        Parameters:
            test_loader (DataLoader).
            session_ids (List[int]): List of session ids of dataset to predict.
            topk (int): top-k labels to predict.
            args (argparse.Namespace).
            store (Store): Get original logs.
            device (str).

        Returns:
            int: Number of anomalies.
        """
        y_pred = {k: {} for k in session_ids}
        real_anomaly = []
        unknown_dict = {}
        list_unknown_idxs = []
        count_unknown = 0
        count_predicted = 0

        progress_bar = tqdm(
            total=len(test_loader),
            desc=f"Predict",
            disable=not self.accelerator.is_local_main_process,
        )
        for batch in test_loader:
            # idxs is a list of indices representing sequences from 0 to len(test_loader)
            idxs = (
                self.accelerator.gather(batch["idx"])
                .detach()
                .clone()
                .cpu()
                .numpy()
                .tolist()
            )

            # sequential is a list of lists containing the indices of the events within a sequence
            sequential = (
                self.accelerator.gather(batch["sequential"])
                .detach()
                .clone()
                .cpu()
                .numpy()
                .tolist()
            )

            # step is a list of lists containing the indices of the events within a sequence
            step = (
                self.accelerator.gather(batch["step"])
                .detach()
                .clone()
                .cpu()
                .numpy()
                .tolist()
            )

            # support label is a list of Boolean values indicating whether each sequence contains an unknown event.
            support_label = (batch["sequential"] >= self.num_classes).any(dim=1)

            support_label = (
                self.accelerator.gather(support_label).cpu().numpy().tolist()
            )

            # batch_label is a list of the next event label for each sequence.
            batch_label = self.accelerator.gather(batch["label"]).cpu().numpy().tolist()

            del batch["idx"]

            with torch.no_grad():
                y = self.accelerator.unwrap_model(self.model).predict_class(
                    batch, top_k=topk, device=device
                )

            # y is a list of lists containing the top-k predicted labels for each sequence.
            y = self.accelerator.gather(y).cpu().numpy().tolist()

            for idx, y_i, b_label, s_label, seq, step in zip(
                idxs, y, batch_label, support_label, sequential, step
            ):
                # y_pred will be e.g {0: {sessionId: step_0:0, step_1: 0, step_2: 0, step_3: 0, step_4: 1}, sessionId+1: {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}, ...} where if step=1 then it is an anomaly
                y_pred[idx][step] = 0
                # if unkown event/S within current sequence
                if s_label == 1:
                    # Note: this code is for writing to a file all unknown events in a chronological order without duplicating logs, using more loops or memory.
                    # See "./testing/unknown.txt for better understanding.

                    # Unknown_dict keeps track of sequences at s steps where unknown events occur, this way dont duplicate logs. so we check real indices as this [[0,1,2,3,4,5,6,7,8,9], [10,11,12,13,14,15,16,17,18,19],[20,21,22,23,24,25,26,27,28,29][...]] and not as [1,2,3,4,5,7,8,9,10], [2,3,4,5,6,7,8,9,10,11],[3,4,4,5,6,7,8,9,10,11,12][...]
                    # if step=1, w=50, h=10, then there are 50 steps.  each 10 steps the step will be reset to 0. steps will look like: [[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[...]]

                    # initial length is declared in each sequence, whereas list_unkown_idxs keeps track of all the unknown events (indices, not sessions ids) for all sequences
                    initial_length = len(list_unknown_idxs)
                    # get the indices where the events are unknown within the current sequence
                    unk_idxs = np.where(np.array(seq) >= self.num_classes)[0]
                    # pseudo = unk_idxs
                    # turn indices to real indices -> step 2 of session [10,20,30,40,50,60,70,80,90,100] with history 5 is [30, 40, 50, 60, 70]. if anomaly in [3] which would be "60" then it is actually the index 5 of the original session
                    unk_idxs = [step + unk_idx for unk_idx in unk_idxs]
                    # create a dictionary with all the session ids where are unknown sequences
                    # print(f"initial list unknown idxs  : {list_unknown_idxs}")
                    if session_ids[idx] not in unknown_dict.keys():
                        unknown_dict[session_ids[idx]] = {}
                        unknown_dict[session_ids[idx]][step] = 1
                        # if there are more than 1 unknown events within a sequence, then append to the list of unknown indices
                        for index in unk_idxs:
                            list_unknown_idxs.append(index)
                    # print(
                    # f"current sequence at {session_ids[idx]} step {step}: {seq} unknown real indices : {unk_idxs} pseudo indices {pseudo}")
                    # print(f"unknown dict: {unknown_dict}")
                    # say this window session of 10 "[10,20,30,40,50,60,70,80,90,100]" with history size of 5 is at step 5 (60,70,80,90,100), real indices must be: [history size + step] = [5,6,7,8,9].
                    if step - args.history_size in unknown_dict[session_ids[idx]]:
                        unknown_dict[session_ids[idx]][step] = 1
                        for index in unk_idxs:
                            list_unknown_idxs.append(index)
                    else:
                        # if a session id already contains unknown event/s and add unknown events
                        if session_ids[idx] in unknown_dict.keys():
                            last_history_indices = list_unknown_idxs[
                                -args.history_size :
                            ]
                            for index in unk_idxs:
                                if index not in last_history_indices:
                                    list_unknown_idxs.append(index)

                    # obtain original session
                    abnormal_session = store.get_test_data(blockId=session_ids[idx])
                    # print(f"Final list unknown idxs  : {list_unknown_idxs}\n")
                    # append only the new unknown events to the file
                    if len(list_unknown_idxs) > initial_length:
                        # print("update")
                        # get all elements that are different from initial length
                        delta_size = len(list_unknown_idxs) - initial_length
                        # slice where unknown events of a session are not yet in the file
                        delta_unknown = list_unknown_idxs[-delta_size:]
                        for index in delta_unknown:
                            # print(f"new anomaly")
                            # new unknown
                            count_unknown += 1
                            real_anomaly.append(
                                {
                                    "EventId": abnormal_session[0]["EventId"][index],
                                    "SEVERITY": abnormal_session[0]["SEVERITY"][index],
                                    "MESSAGE": abnormal_session[0]["MESSAGE"][index],
                                    "_zl_timestamp": abnormal_session[0][
                                        "_zl_timestamp"
                                    ][index],
                                    "log_uuid": abnormal_session[0]["log_uuid"][index],
                                }
                            )
                        # print(f"\n")
                else:
                    y_pred[idx][step] = 0 | (b_label not in y_i)
                    # if the next event is not in the top-k predictions, then last element from current sequence at current step is an anomaly
                    # keep track step by step for each log in a w-session. We are one step ahead of the current step, this is because anomalies are labeled in the current step, so we need to check the previous step.
                    if step >= 1 and y_pred[idx][step - 1] == 1:
                        # obtain original session
                        abnormal_session = store.get_test_data(blockId=session_ids[idx])
                        count_predicted += 1
                        # get the exact predicted anomaly within the original session
                        real_anomaly.append(
                            {
                                "EventId": abnormal_session[0]["EventId"][
                                    args.history_size + step - 1
                                ],
                                "SEVERITY": abnormal_session[0]["SEVERITY"][
                                    args.history_size + step - 1
                                ],
                                "MESSAGE": abnormal_session[0]["MESSAGE"][
                                    args.history_size + step - 1
                                ],
                                "_zl_timestamp": abnormal_session[0]["_zl_timestamp"][
                                    args.history_size + step - 1
                                ],
                                "log_uuid": abnormal_session[0]["log_uuid"][
                                    args.history_size + step - 1
                                ],
                            }
                        )
            progress_bar.update(1)

        progress_bar.close()
        self.logger.info(f"Computing metrics...")
        # creating file with all anomalies
        test_folder = f"./testing/{self.args.dataset_folder}/{self.args.model_name}"
        # remove slashes
        dataset_folder = self.args.dataset_folder.replace("/", "")
        current_date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        filename = f"{current_date}_anomalies_w{self.args.window_size}_s{self.args.step_size}_his{self.args.history_size}_train_{self.args.train_size}_topk{self.topk}.txt"
        os.makedirs(test_folder, exist_ok=True)
        # write to file
        with open(os.path.join(test_folder, filename), "w") as f:
            for log in real_anomaly:
                f.write(f"{log}\n")

        progress_bar.close()
        return evaluate_predictions(count_unknown, count_predicted)

    def load_model(self, model_path: str) -> None:
        """
        Description:
            Load model from a checkpoint.

        Parameters:
            model_path (str): Path to model checkpoint.
        """
        checkpoint = torch.load(model_path)
        self.model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(checkpoint["model"])
        self.model = self.accelerator.prepare(self.model)
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def start_predicting(self) -> None:
        """
        Predicting model on test dataset.
        """

        self.logger.info(
            f"Start predicting {self.model_name} model on {self.device} device with top-{self.topk} recommendation"
        )
        self.load_model(f"{self.args.output_dir}/models/{self.model_name}.pt")
        print(f"vocab size: {self.num_classes}")

        if self.topk > self.num_classes:
            self.topk = self.num_classes - 1
            self.logger.info(
                f"Selected topk is bigger than vocab size, setting topk to automatically be lower than vocab size: {self.num_classes}."
            )

        self.predict_unsupervised(
            dataset=self.test_dataset,
            topk=self.topk,
            device=self.device,
            session_ids=self.session_ids,
            args=self.args,
            store=self.store,
        )
