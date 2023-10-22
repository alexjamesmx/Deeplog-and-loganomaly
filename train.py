import numpy as np
import os
import torch
import argparse

from sklearn.metrics import accuracy_score, top_k_accuracy_score
from tqdm import tqdm
from transformers import get_scheduler
from torch.utils.data import DataLoader
from typing import Any,  List, Union, Tuple

from data.store import Store
from data.dataset import LogDataset
from utils.helpers import get_optimizer

from data.process import process_sessions, create_datasets


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 args: argparse.Namespace,
                 vocabs,
                 store,
                 ):
        # constant values
        self.accelerator = args.accelerator
        self.device = self.accelerator.device
        self.model = model.to(self.device)
        self.optimizer = get_optimizer(args, self.model.parameters())
        self.is_train = True
        self.no_epochs = args.max_epoch
        self.scheduler_type = args.scheduler
        self.logger = args.logger
        self.num_classes = len(vocabs)
        self.save_dir = args.save_dir
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.warmup_rate = args.warmup_rate
        self.accumulation_step = args.accumulation_step
        # dynamic values
        self.topk = args.topk
        self.store = store
        self.scheduler = None

        self.train_sessions, self.valid_sessions = process_sessions(
            path=args.train_path,
            args=args,
            is_train=self.is_train,
            store=store,
            logger=self.logger)

        # print(self.train_sessions)

        self.train_dataset, self.valid_dataset, self.train_parameters, self.valid_parameters = create_datasets(
            train_data=self.train_sessions,
            valid_data=self.valid_sessions,
            vocab=vocabs,
            args=args,
            is_train=self.is_train,
            store=store,
            logger=self.logger,
        )

        self.valid_session_ids = self.valid_dataset.get_session_ids()
        print(vocabs.stoi)

    def _train_epoch(self,
                     train_loader: DataLoader,
                     device: str,
                     scheduler: Any,
                     progress_bar: Any):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0
        for idx, batch in enumerate(train_loader):
            # print(batch)

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
        for batch in val_loader:
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
            # if self.logger is not None:
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
        return find_topk(test_loader)

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

    def start_training(self):
        train_loss, val_loss, val_acc, topk = self.train(
            device=self.device,
            save_dir=self.save_dir,
            model_name=self.model_name,
            topk=self.topk
        )
        self.logger.info(
            f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        minimum_recommended_topk = self.predict_unsupervised(self.valid_dataset,
                                                             self.valid_session_ids,
                                                             topk=self.topk,
                                                             device=self.device,
                                                             is_valid=True)

        self.logger.info(
            f"Top-{self.topk} Min recommendation: {minimum_recommended_topk}\n")

    # not in usage, but cam be implemented
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
