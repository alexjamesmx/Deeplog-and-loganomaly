import numpy as np
import os
import torch
import logging
import argparse

from sklearn.metrics import accuracy_score, top_k_accuracy_score
from tqdm import tqdm
from transformers import get_scheduler
from torch.utils.data import DataLoader
from typing import Any, Optional, List, Union, Tuple

from data.store import Store
from data.dataset import LogDataset
from utils.helpers import evaluate_predictions


class Trainer:
    def __init__(self, model,
                 train_dataset: LogDataset = None,
                 valid_dataset: LogDataset = None,
                 is_train: bool = True,
                 optimizer: torch.optim.Optimizer = None,
                 no_epochs: int = 100,
                 batch_size: int = 32,
                 scheduler_type: str = 'linear',
                 warmup_rate: float = 0.1,
                 accumulation_step: int = 1,
                 logger: logging.Logger = None,
                 accelerator: Any = None,
                 num_classes: int = 2,
                 ):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.is_train = is_train
        self.optimizer = optimizer
        self.no_epochs = no_epochs
        self.batch_size = batch_size
        self.scheduler_type = scheduler_type
        self.warmup_rate = warmup_rate
        self.accumulation_step = accumulation_step
        self.logger = logger
        self.accelerator = accelerator
        self.num_classes = num_classes
        self.scheduler = None
        self.original_anomalies = None

    def _train_epoch(self,
                     train_loader: DataLoader,
                     device: str,
                     scheduler: Any,
                     progress_bar: Any):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0
        for idx, batch in enumerate(train_loader):

            outputs = self.model(batch, device=device)
            loss = outputs.loss
            total_loss += loss.item()
            loss = loss / self.accumulation_step
            self.accelerator.backward(loss)
            if (idx + 1) % self.accumulation_step == 0 or idx == len(train_loader) - 1:
                self.optimizer.step()
                self.optimizer.zero_grad()
                scheduler.step()
                progress_bar.update(1)
                progress_bar.set_postfix({'loss': total_loss / (idx + 1)})

        return total_loss / len(train_loader)

    def _valid_epoch(self,
                     val_loader: DataLoader,
                     device: str,
                     topk: int = 1):
        self.model.eval()
        y_pred = []
        y_true = []
        losses = []
        for idx, batch in enumerate(val_loader):
            del batch['idx']
            with torch.no_grad():
                outputs = self.model(batch, device=device)
            loss = outputs.loss
            probabilities = self.accelerator.gather(outputs.probabilities)
            y_pred.append(probabilities.detach().clone().cpu().numpy())
            losses.append(loss.item())
            label = self.accelerator.gather(batch['label'])
            y_true.append(label.detach().clone().cpu().numpy())
        # concatenate because there are arrays of arrays
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        loss = np.mean(losses)
        if topk > 1:
            for k in range(1, self.num_classes + 1):
                acc = top_k_accuracy_score(
                    y_true, y_pred, k=k, labels=np.arange(self.num_classes))
                if acc >= 0.997:
                    return loss, acc, k
        else:
            acc = accuracy_score(y_true, np.argmax(y_pred, axis=1))
        return loss, acc, topk

    def train(self,
              device: str = 'cpu',
              save_dir: str = None,
              model_name: str = None,
              topk: int = 9):

        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(
            self.valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.model.to(device)
        self.model, self.optimizer, train_loader, val_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader
        )
        # Compute the number of training steps
        num_training_steps = int(
            self.no_epochs * len(train_loader) / self.accumulation_step)
        # Compute the number of warmup steps
        num_warmup_steps = int(num_training_steps * self.warmup_rate)
        # Create the scheduler
        self.scheduler = get_scheduler(
            self.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        # Create a progress bar
        progress_bar = tqdm(range(num_training_steps), desc=f"Training",
                            disable=not self.accelerator.is_local_main_process)
        total_train_loss = 0
        total_val_loss = 0
        total_val_acc = 0
        # Train the model
        for epoch in range(self.no_epochs):
            # Train the model for one epoch
            train_loss = self._train_epoch(
                train_loader, device, self.scheduler, progress_bar)
            # Evaluate the model on the validation set
            val_loss, val_acc, valid_k = self._valid_epoch(
                val_loader, device, topk=topk)
        #     self.logger.info(
        #         f"Epoch {epoch + 1}||Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            total_train_loss += train_loss
            total_val_loss += val_loss
            total_val_acc += val_acc
            # Save the model
            if save_dir is not None and model_name is not None:
                self.save_model(save_dir, model_name)
        _, _, train_k = self._valid_epoch(train_loader, device, topk=topk)
        progress_bar.close()
        self.logger.info(
            f"Train top-{topk}: {train_k}, Valid top-{topk}: {valid_k}")
        return total_train_loss / self.no_epochs, val_loss, val_acc, max(train_k, valid_k)

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
        normal, anomalies = evaluate_predictions(y_pred)
        return normal, anomalies

    # not in usage, but cam be later implemented
    # def train_on_false_positive(self,
    #                             false_positive_dataset: LogDataset,
    #                             device: str = 'cpu',
    #                             save_dir: str = None,
    #                             model_name: str = None,
    #                             topk: int = 9):
    #     # NOTE shuffle was true
    #     train_loader = DataLoader(
    #         false_positive_dataset, batch_size=self.batch_size, shuffle=False)
    #     self.model.to(device)
    #     self.model, train_loader = self.accelerator.prepare(
    #         self.model, train_loader)
    #     self.optimizer.zero_grad()
    #     # Train the model on the false positive anomaly data
    #     num_training_steps = int(
    #         self.no_epochs * len(train_loader) / self.accumulation_step)
    #     num_warmup_steps = int(num_training_steps * self.warmup_rate)
    #     self.scheduler = get_scheduler(
    #         self.scheduler_type,
    #         optimizer=self.optimizer,
    #         num_warmup_steps=num_warmup_steps,
    #         num_training_steps=num_training_steps
    #     )
    #     progress_bar = tqdm(range(num_training_steps), desc=f"Training",
    #                         disable=not self.accelerator.is_local_main_process)
    #     total_train_loss = 0

    #     for epoch in range(self.no_epochs):
    #         train_loss = self._train_epoch(
    #             train_loader, device, self.scheduler, progress_bar)
    #         total_train_loss += train_loss
    #     if save_dir is not None and model_name is not None:
    #         self.save_model(save_dir, model_name)
    #     _, _, train_k = self._valid_epoch(train_loader, device, topk=topk)
    #     print(
    #         f"total_train_loss: {total_train_loss / self.no_epochs} top-{topk}: {train_k}")
    #     return total_train_loss / self.no_epochs, train_k

    def save_model(self, save_dir: str, model_name: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save(
            {
                "lr_scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "optimizer": self.optimizer.state_dict(),
                "model": self.model.state_dict()
            },
            f"{save_dir}/{model_name}.pt"
        )

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path)
        self.model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(checkpoint['model'])
        self.model = self.accelerator.prepare(self.model)
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
