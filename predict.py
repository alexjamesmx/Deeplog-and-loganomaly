import torch
import numpy as np
import os
import argparse

from data.process import process_sessions, create_datasets
from data.dataset import LogDataset

from utils.helpers import get_optimizer
from utils.helpers import evaluate_predictions

from data.store import Store
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Tuple, Union


class Predicter():

    def __init__(self,
                 model,
                 vocabs,
                 args,
                 store):
        # constant values
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
        self.logger = args.logger
        self.num_classes = len(vocabs)
        self.model_name = args.model_name
        self.args = args
        # dynamic values
        self.topk = args.topk
        self.store = store
        self.scheduler = None
        print(vocabs.stoi)

        self.test_sessions = process_sessions(
            path=args.test_path,
            args=args,
            is_train=self.is_train,
            store=store,
            logger=self.logger)

        self.test_dataset,  self.test_parameters = create_datasets(
            test_data=self.test_sessions,
            vocab=vocabs,
            args=args,
            is_train=self.is_train,
            store=store,
            logger=self.logger,
        )
        self.session_ids = self.test_dataset.get_session_ids()

    def predict_unsupervised(self,
                             dataset: LogDataset,
                             session_ids: List[int],
                             topk: int,
                             device: str = 'cpu',
                             is_valid: bool = False,
                             args:  argparse.Namespace = None,
                             store: Store = None
                             ) -> Union[Tuple[float, float, float, float], Tuple[float, int]]:

        test_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False)
        self.model.to(device)
        self.model, test_loader = self.accelerator.prepare(
            self.model, test_loader)
        self.model.eval()

        return self.predict_unsupervised_helper(
            test_loader=test_loader,
            topk=topk,
            session_ids=session_ids,
            args=args,
            store=store,
            device=device)

    def predict_unsupervised_helper(self,
                                    test_loader,
                                    session_ids: List,
                                    topk: int,
                                    args: argparse.Namespace = None,
                                    store=Store,
                                    device: str = 'cpu',
                                    ) -> Tuple[float, float, float, float]:
        y_pred = {k: {} for k in session_ids}

        exact_predicted_anomaly = []
        unknown_dict = {}
        list_unknown_idxs = []
        count_unknown = 0
        count_predicted = 0

        # abnormal_sessions = []
        progress_bar = tqdm(total=len(test_loader), desc=f"Predict",
                            disable=not self.accelerator.is_local_main_process)
        for batch in test_loader:

            idxs = self.accelerator.gather(
                batch['idx']).detach().clone().cpu().numpy().tolist()
            sequential = self.accelerator.gather(
                batch["sequential"]).detach().clone().cpu().numpy().tolist()

            step = self.accelerator.gather(
                batch["step"]).detach().clone().cpu().numpy().tolist()

            support_label = (batch['sequential'] >=
                             self.num_classes).any(dim=1)

            support_label = self.accelerator.gather(
                support_label).cpu().numpy().tolist()
            batch_label = self.accelerator.gather(
                batch['label']).cpu().numpy().tolist()

            del batch['idx']

            with torch.no_grad():
                y = self.accelerator.unwrap_model(self.model).predict_class(
                    batch, top_k=topk, device=device)
            y = self.accelerator.gather(y).cpu().numpy().tolist()

            # idxs is a list of indices representing sequences
            # y is a list of lists containing the top-k predicted labels for sequential.
            # batch_label is a list that contains the next event label for each batch.
            # support_label is a list of Boolean values indicating whether each sequence contains an unknown event.
            for idx, y_i, b_label, s_label, seq, step in zip(idxs, y, batch_label, support_label, sequential, step):
                # y_pred will be e.g {0: {sessionId: 0:0, 1: 0, 2: 0, 3: 0, 4: 1}, sessionId+1: {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}, ...}
                y_pred[idx][step] = 0
                # if a unkown events within current sequence appears
                if s_label == 1:
                    # Note: this can be refactored, this code is for writing to a file all unknown events without duplicating logs and loops in a chronological order
                    # this code also finds the exact original logs where unknown events appear
                    # It keeps track of changes so sessions where there are several steps with several unknown events dont get duplicated.

                    # See "./testing/unknown.txt for better understanding.
                    # Unknown_dict purpose is to keep track of h sequences at steps where unknown events occur, this way dont duplicate logs. 1 here serves as a pointer of existent unique sequences. so we check real indices as this [[0,1,2,3,4,5,6,7,8,9], [10,11,12,13,14,15,16,17,18,19],[...]] and not check as [1,2,3,4,5,7,8,9,10], [2,3,4,5,6,7,8,9,10,11],[3,4,4,5,6,7,8,9,10,11,12][...]
                    # this way, as steps reset each h size, see -> if step=1, w=50, h=10, then there are 50 steps. and each 10 steps, the step will be reset to 0. steps will look like: [[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[...]]
                    # Unknown_dict E.g {sessionId: {0:1,10:1,20:1,30:1,40:1,}}

                    # initial length is declared in each sequence, whereas list_unkown_idxs keeps track of all the unknown events (indices, not sessions ids) for all sequences

                    initial_length = len(list_unknown_idxs)
                    # get the indices where the events are unknown within the current sequence
                    unk_idxs = np.where(np.array(seq) >= self.num_classes)[0]
                    # pseudo = unk_idxs
                    # turn indices to real indices -> step 1 of session [10,20,30,40,50,60,70,80,90,100] with history 5 is [20,30,40,50,60]. if anomaly in [3] = 50 then it is actually the index 5 of the original session
                    unk_idxs = [step + unk_idx for unk_idx in unk_idxs]
                    # this if is only executed when a unknown event is not in unknown_dict
                    # create a dictionary with all the session ids where are unknown sequences
                    # print(f"initial list unknown idxs  : {list_unknown_idxs}")

                    if session_ids[idx] not in unknown_dict.keys():
                        # {sessionId: {},...,  sesionId: {}, }
                        unknown_dict[session_ids[idx]] = {}
                        # label current step as abnormal
                        # if sequence [10,20,30,40,50,60,70,80,90,100]
                        unknown_dict[session_ids[idx]][step] = 1
                        # if there are more than 1 unknown events within a sequence, then append to the list of unknown indices
                        for index in unk_idxs:
                            list_unknown_idxs.append(index)
                    # print(
                        # f"current sequence at {session_ids[idx]} step {step}: {seq} unknown real indices : {unk_idxs} pseudo indices {pseudo}")
                    # print(f"unknown dict: {unknown_dict}")
                    # if a unknown event or several unknown events in a w-session of 50 with history 10 e.g [[0],[...],[10,20,30,40,50,60,70,80,90,100],[...],[w-1]] then we obtain real indices, but each index is a different step, so
                    # say this sequence [10,20,30,40,50,60,70,80,90,100] is at step 5, real indices must be: [history size + step] = [5,6,7,8,9,10,11,12,13,14].
                    if step-args.history_size in unknown_dict[session_ids[idx]]:
                        unknown_dict[session_ids[idx]][step] = 1
                        for index in unk_idxs:
                            list_unknown_idxs.append(index)
                    else:
                        # for all unknown events in a history sequence check
                        # if a session id already contains unknown event/s and add unknown events

                        if session_ids[idx] in unknown_dict.keys():
                            last_history_indices = list_unknown_idxs[-args.history_size:]
                            for index in unk_idxs:
                                # if index not in last history size of list_unknown_idxs
                                # if index not in list_unknown_idxs[-args.history_size:]:
                                if index not in last_history_indices:

                                    list_unknown_idxs.append(index)
                    # obtain original session
                    abnormal_session = store.get_test_data(
                        blockId=session_ids[idx])
                    # if there were changes
                    # print(f"Final list unknown idxs  : {list_unknown_idxs}\n")
                    if len(list_unknown_idxs) > initial_length:
                        # print("update")
                        # get all elements that are different from initial map
                        delta_size = len(list_unknown_idxs) - \
                            initial_length
                        # slice where unknown events of a session are not yet in the file
                        delta_unknown = list_unknown_idxs[-delta_size:]
                        for index in delta_unknown:
                            # print(f"new anomaly")
                            # new unknown
                            count_unknown += 1
                            exact_predicted_anomaly.append(
                                {
                                    "EventId": abnormal_session[0]["EventId"][index],
                                    "SEVERITY": abnormal_session[0]["SEVERITY"][index],
                                    "MESSAGE": abnormal_session[0]["MESSAGE"][index],
                                    "_zl_timestamp": abnormal_session[0]["_zl_timestamp"][index],
                                    "log_uuid": abnormal_session[0]["log_uuid"][index],
                                }
                            )
                        # print(f"\n")

                else:
                    # if the next event is not in the top-k predictions, then current sequence is anomalous
                    y_pred[idx][step] = y_pred[idx][step] | (
                        b_label not in y_i)
                    # keep track step by step for each log in a w-session. We are one step ahead of the current step, this is because anomalies are labeled in the current step, so we need to check the previous step.
                    if step >= 1 and y_pred[idx][step-1] >= 1:
                        # obtain original session
                        abnormal_session = store.get_test_data(
                            blockId=session_ids[idx])
                        count_predicted += 1
                        # get the exact predicted anomaly within the original session
                        exact_predicted_anomaly.append(
                            {
                                "EventId": abnormal_session[0]["EventId"][args.history_size + step - 1],
                                "SEVERITY": abnormal_session[0]["SEVERITY"][args.history_size + step - 1],
                                "MESSAGE": abnormal_session[0]["MESSAGE"][args.history_size + step-1],
                                "_zl_timestamp": abnormal_session[0]["_zl_timestamp"][args.history_size + step-1],
                                "log_uuid": abnormal_session[0]["log_uuid"][args.history_size + step-1],
                            }
                        )
            progress_bar.update(1)
        progress_bar.close()
        idxs = list(y_pred.keys())
        self.logger.info(f"Computing metrics...")

        y_pred = [y_pred[idx] for idx in idxs]

        test_folder = f"./testing/{self.args.dataset_folder}"
        # remove slashes
        dataset_folder = self.args.dataset_folder.replace("/", "")
        filename = f"{dataset_folder}_predicted_w{self.args.window_size}_s{self.args.step_size}train_{self.args.train_size}_topk{self.topk}_his{self.args.history_size}.txt"
        os.makedirs(test_folder, exist_ok=True)
        with open(os.path.join(test_folder, filename), 'w') as f:
            for log in exact_predicted_anomaly:
                f.write(f"{log}\n")

        progress_bar.close()
        return evaluate_predictions(count_unknown, count_predicted)

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path)
        self.model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(checkpoint['model'])
        self.model = self.accelerator.prepare(self.model)
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def start_predicting(self):
        self.logger.info(
            f"Start predicting {self.model_name} model on {self.device} device with top-{self.topk} recommendation")
        self.load_model(f"{self.args.output_dir}/models/{self.model_name}.pt")
        print(f"vocab size: {self.num_classes}")

        if (self.topk > self.num_classes):
            self.topk = self.num_classes - 10
            self.logger.info(
                f"topk is bigger than vocab size, setting topk automatically lower to {self.num_classes}.")

        self.predict_unsupervised(dataset=self.test_dataset,
                                  topk=self.topk,
                                  device=self.device,
                                  is_valid=self.is_train,
                                  session_ids=self.session_ids,
                                  args=self.args,
                                  store=self.store,
                                  )
