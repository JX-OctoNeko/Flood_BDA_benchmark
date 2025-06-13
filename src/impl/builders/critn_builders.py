# Custom criterion builders

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.misc import CRITNS

@CRITNS.register_func('WBCE_critn')
def build_weighted_bce_critn(C):
    # assert len(C['weights']) == 2
    # pos_weight = C['weights'][1]/C['weights'][0]
    # return nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight])) # 原代码
    return nn.CrossEntropyLoss(weight=torch.Tensor(C['weights'])) 
