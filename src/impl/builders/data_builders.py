# Custom data builders

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

import constants
from utils.data_utils.augmentations import *
from utils.data_utils.preprocessors import *
from core.misc import DATA, R
from core.data import (
    build_train_dataloader, build_eval_dataloader, get_common_train_configs, get_common_eval_configs
)
        

@DATA.register_func('XVIEW_train_dataset')
def build_xview_train_dataset(C):
    configs = get_common_train_configs(C)
    configs.update(dict(
        transforms=(Choose(
            HorizontalFlip(), VerticalFlip(),
            Rotate('90'), Rotate('180'), Rotate('270'),
            Shift(),
            Identity()), Normalize(np.asarray(C['mu']), np.asarray(C['sigma'])), None),
        root=constants.IMDB_XVIEW,
    ))

    from data.xview import XVIEWDataset
    return build_train_dataloader(XVIEWDataset, configs, C)


@DATA.register_func('XVIEW_eval_dataset')
def build_xview_eval_dataset(C):
    configs = get_common_eval_configs(C)
    configs.update(dict(
        transforms=(None, Normalize(np.asarray(C['mu']), np.asarray(C['sigma'])), None),
        root=constants.IMDB_XVIEW,
    ))

    from data.xview import XVIEWDataset
    return DataLoader(
        XVIEWDataset(**configs),
        batch_size=C['batch_size'],
        shuffle=False,
        num_workers=C['num_workers'],
        drop_last=False,
        pin_memory=C['device']!='cpu'
    )
