# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset


class LogDataset(Dataset):
    def __init__(self,
                 sequentials=None,
                 quantitatives=None,
                 semantics=None,
                 labels=None,
                 sequentials_idxs=None,
                 session_ids=None,
                 steps=None):
        if sequentials is None and quantitatives is None and semantics is None:
            raise ValueError('Provide at least one feature type')
        self.sequentials = sequentials
        self.quantitatives = quantitatives
        self.semantics = semantics
        self.labels = labels
        self.sequentials_idxs = sequentials_idxs
        self.session_ids = session_ids
        self.steps = steps

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {'label': self.labels[idx], 'idx': self.sequentials_idxs[idx]}
        if self.steps is not None:
            item['step'] = self.steps[idx]
        if self.sequentials is not None:
            item['sequential'] = torch.from_numpy(
                np.array(self.sequentials[idx]))
        if self.semantics is not None:
            item['semantic'] = torch.from_numpy(
                np.array(self.semantics[idx])).float()
        return item

    def get_sequential(self):
        return self.sequentials

    def get_quantitative(self):
        return self.quantitatives

    def get_semantic(self):
        return self.semantics

    def get_label(self):
        return self.labels

    def get_idx(self):
        return self.sequentials_idxs

    def get_session_ids(self):
        return self.session_ids

    # def get_shape(self):
    #     return {
    #         'sequential': len(self.sequentials),
    #         'semantic': len(self.semantics),
    #         'quantitative': len(self.quantitatives),
    #         'label': len(self.labels),
    #         'idx': len(self.sequentials_idxs)
    #     }
