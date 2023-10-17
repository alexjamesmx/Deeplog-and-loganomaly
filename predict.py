from utils.helpers import get_optimizer
from data.preprocess import preprocess_data, preprocess_slidings
from data.dataset import LogDataset
from typing import List, Tuple, Union
import argparse
from data.store import Store
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.helpers import evaluate_predictions


class Predicter():

    def __init__(self,
                 model,
                 args,
                 logger,
                 store,
                 vocabs):
        self.model = model
        self.is_train = False
        self.no_epochs = args.max_epoch
        self.batch_size = args.batch_size
        self.scheduler_type = args.scheduler
        self.warmup_rate = args.warmup_rate
        self.accumulation_step = args.accumulation_step
        self.logger = logger
        self.accelerator = args.accelerator
        self.optimizer = get_optimizer(args, self.model.parameters())
        self.num_classes = len(vocabs)
        self.device = self.accelerator.device
        self.model_name = args.model_name
        self.topk = args.topk
        self.args = args
        self.store = store
        self.scheduler = None

        self.test_sessions = preprocess_data(
            path=args.test_path,
            args=args,
            is_train=self.is_train,
            store=store,
            logger=logger)

        self.test_dataset,  self.test_parameters = preprocess_slidings(
            test_data=self.test_sessions,
            vocab=vocabs,
            args=args,
            is_train=self.is_train,
            store=store,
            logger=logger,
        )
        self.session_ids = self.test_dataset.get_session_labels()

    def predict_unsupervised(self,
                             dataset: LogDataset,
                             session_ids: List[int],
                             topk: int,
                             device: str = 'cpu',
                             is_valid: bool = False,
                             args:  argparse.Namespace = None,
                             store: Store = None
                             ) -> Union[Tuple[float, float, float, float], Tuple[float, int]]:
        def find_topk(dataloader):
            y_topk = []
            torch.set_printoptions(threshold=torch.inf)
            for batch in dataloader:
                # Get the label for the current batch.
                label = self.accelerator.gather(batch['label'])
                with torch.no_grad():
                    # here not using topk because we want to find the position of the label in the array
                    y_prob = self.model.predict(batch, device=device)
                # get the positions of the probabilities in descending order
                y_pred = torch.argsort(y_prob, dim=1, descending=True)
                # Get the positions where the predictions match the label.
                y_pos = torch.where(y_pred == label.unsqueeze(1))[1] + 1
                # Add the positions of the labels to the list of top-k values.
                y_topk.extend(y_pos.cpu().numpy().tolist())
                # The top-k values for all batches are then added to a list and the 99th percentile of the top-k values is calculated. This value is the top-k value for which the label is in the top-k most likely outcomes.
                minimum_recommended_topk = int(
                    np.ceil(np.percentile(y_topk, 0.99)))
            return minimum_recommended_topk

        test_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False)
        self.model.to(device)
        self.model, test_loader = self.accelerator.prepare(
            self.model, test_loader)
        self.model.eval()
        if is_valid:
            return find_topk(test_loader)
        else:
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

        original_anomalies_predicted = []
        exact_predicted_anomaly = []
        exact_unkown_anomaly = []
        unknown_dict = {}
        map_unknown_idxs = []
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
                y_pred[idx][step] = 0
                unk_idxs = np.where(np.array(seq) >= self.num_classes)[0]
                if s_label == 1:
                    initial_map_length = len(map_unknown_idxs)
                    unk_idxs = np.where(np.array(seq) >= self.num_classes)[0]
                    unk_idxs = [step + unk_idx for unk_idx in unk_idxs]

                    if session_ids[idx]not in unknown_dict.keys():
                        unknown_dict[session_ids[idx]] = {}
                        unknown_dict[session_ids[idx]][step] = 1
                        for element in unk_idxs:
                            map_unknown_idxs.append(element)
                    if step-args.history_size in unknown_dict[session_ids[idx]]:
                        unknown_dict[session_ids[idx]][step] = 1
                        for element in unk_idxs:
                            map_unknown_idxs.append(element)
                    else:
                        if session_ids[idx] in unknown_dict.keys():
                            for element in unk_idxs:
                                # if element not in last history size of map unknown idxs
                                if element not in map_unknown_idxs[-args.history_size:]:
                                    map_unknown_idxs.append(element)

                    abnormal_session = store.get_test_data(
                        blockId=session_ids[idx])

                    if len(map_unknown_idxs) > initial_map_length:
                        # get all elements that are in map_unknown_idxs but not in initial_map without set
                        size_diff = len(map_unknown_idxs) - initial_map_length
                        diff_map = map_unknown_idxs[-size_diff:]

                        for unk_idx in diff_map:
                            exact_predicted_anomaly.append(
                                {
                                    # "SessionId": abnormal_session[0]["SessionId"],
                                    # "real_idx": unk_idx,
                                    # "step": step,
                                    "EventId": abnormal_session[0]["EventId"][unk_idx],
                                    "SEVERITY": abnormal_session[0]["SEVERITY"][unk_idx],
                                    "MESSAGE": abnormal_session[0]["MESSAGE"][unk_idx],
                                    "_zl_timestamp": abnormal_session[0]["_zl_timestamp"][unk_idx],
                                    "log_uuid": abnormal_session[0]["log_uuid"][unk_idx],
                                }
                            )
                else:
                    y_pred[idx][step] = y_pred[idx][step] | (
                        b_label not in y_i)
                    if step > 0 and y_pred[idx][step-1] > 0:
                        abnormal_session = store.get_test_data(
                            blockId=session_ids[idx])
                        if y_pred[idx][step-1] == 1:
                            original_anomalies_predicted.append(
                                abnormal_session)
                            exact_predicted_anomaly.append(
                                {
                                    # "SessionId": abnormal_session[0]["SessionId"],
                                    "EventId": abnormal_session[0]["EventId"][args.history_size + step],
                                    "SEVERITY": abnormal_session[0]["SEVERITY"][args.history_size + step],
                                    "MESSAGE": abnormal_session[0]["MESSAGE"][args.history_size + step],
                                    "_zl_timestamp": abnormal_session[0]["_zl_timestamp"][args.history_size + step],
                                    "log_uuid": abnormal_session[0]["log_uuid"][args.history_size + step],
                                }
                            )
            progress_bar.update(1)
        progress_bar.close()
        idxs = list(y_pred.keys())
        self.logger.info(f"Computing metrics...")

        y_pred = [y_pred[idx] for idx in idxs]

        with open("./testing/predicted_w100_s100_t0.3_topk9_hs_20.txt", "w") as f:
            for log in exact_predicted_anomaly:
                f.write(f"{log}\n\n")
        with open("./testing/unkown.txt", "w") as f:
            for log in exact_unkown_anomaly:
                f.write(f"{log}\n\n")

        progress_bar.close()
        return evaluate_predictions(y_pred)

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

        normal, anomalies = self.predict_unsupervised(dataset=self.test_dataset,
                                                      topk=self.topk,
                                                      device=self.device,
                                                      is_valid=self.is_train,
                                                      session_ids=self.session_ids,
                                                      args=self.args,
                                                      store=self.store,
                                                      )

        self.logger.info(
            f"Normal: {normal} - Anomalies: {anomalies}")
