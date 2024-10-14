import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedLoss(nn.Module):
    def __init__(self, critns, coeffs=1.0):
        super().__init__()
        self.critns = critns
        if isinstance(coeffs, float):
            coeffs = [coeffs]*len(critns)
        if len(coeffs) != len(critns):
            raise ValueError
        self.coeffs = coeffs

    def forward(self, pred, tar):
        loss = 0.0
        for critn, coeff in zip(self.critns, self.coeffs):
            loss += coeff * critn(pred, tar)
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, critn, coeffs=1.0):
        super().__init__()
        self.critn = critn
        self.coeffs = coeffs

    def forward(self, preds, tar):
        if isinstance(self.coeffs, float):
            coeffs = [self.coeffs]*len(preds)
        else:
            coeffs = self.coeffs
        if len(coeffs) != len(preds):
            # print(self.coeffs)
            # print(len(coeffs))
            # print(len(preds))
            # print(type(preds))
            # print(preds.shape)
            raise ValueError
        loss = 0.0
        for coeff, pred in zip(coeffs, preds):
            # print(type(coeff))
            # print(type(pred))
            # print(type(tar))
            # print(pred.dtype)
            # print(tar.dtype)
            tar = tar.long()
            loss += coeff * self.critn(pred, tar)
        return loss


class CombinedLoss_DS(nn.Module):
    def __init__(self, critn_main, critn_aux, coeff_main=1.0, coeffs_aux=1.0, main_idx=0):
        super().__init__()
        self.critn_main = critn_main
        self.critn_aux = critn_aux
        self.coeff_main = coeff_main
        self.coeffs_aux = coeffs_aux
        self.main_idx = main_idx

    def forward(self, preds, tar, tar_aux=None):
        if tar_aux is None:
            tar_aux = tar

        pred_main = preds[self.main_idx]
        preds_aux = [pred for i, pred in enumerate(preds) if i != self.main_idx]

        if isinstance(self.coeffs_aux, float):
            coeffs_aux = [self.coeffs_aux]*len(preds_aux)
        else:
            coeffs_aux = self.coeffs_aux
        if len(coeffs_aux) != len(preds_aux):
            raise ValueError

        loss = self.critn_main(pred_main, tar)
        for coeff, pred in zip(coeffs_aux, preds_aux):
            loss += coeff * self.critn_aux(pred, tar_aux)
        return loss


# Refer to https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2):
        super().__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, pred, tar):
        pred, tar = pred.flatten(1), tar.flatten(1)
        prob = F.sigmoid(pred)

        num = (prob*tar).sum(1) + self.smooth
        den = (prob.pow(self.p) + tar.pow(self.p)).sum(1) + self.smooth

        loss = 1 - num/den
        
        return loss.mean()

class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2):
        super().__init__()
        self.smooth = smooth
        self.p = p

    def one_hot(self, input, num_classes):
        return torch.eye(num_classes)[input]

    def forward(self, pred, tar):
        # 将 tar 转换为 one-hot 编码
        targets_one_hot = self.one_hot(tar.long(), pred.shape[1]).float()

        # 计算预测概率
        probas = F.softmax(pred, dim=1)

        # 对于每个类别，分别计算 Dice 损失
        dice_loss_per_class = []
        for c in range(pred.shape[1]):
            pred_c = probas[:, c:c+1]
            target_c = targets_one_hot[:, c:c+1]

            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

            dice_loss_per_class.append(1 - (2.0 * intersection + self.smooth) / (union + self.smooth))

        # 计算所有类别的 Dice 损失均值
        dice_loss = torch.stack(dice_loss_per_class).mean()

        return dice_loss

class BCLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.m = margin
        self.eps = 1e-4

    def forward(self, pred, tar):
        # 为了sta和dsamnet修改代码
        if pred.size(1) >= tar.size(0):
            pred = pred[:, :tar.size(0), :, :]
        else:
            tmp = torch.zeros([pred.size(0), tar.size(0), pred.size(2), pred.size(3)], device=pred.device)
            tmp[:, pred.size(1), :, :] = pred
            pred = tmp


        utar = 1-tar
        n_u = utar.sum() + self.eps
        n_c = tar.sum() + self.eps
        loss = 0.5*torch.sum(utar*torch.pow(pred, 2)) / n_u + \
            0.5*torch.sum(tar*torch.pow(torch.clamp(self.m-pred, min=0.0), 2)) / n_c
        return loss

class MultiClassBCELoss(nn.Module):
    def __init__(self, margin=2.0, reduction='mean', weights=None):
        super().__init__()
        self.m = margin
        self.eps = 1e-4
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(pos_weight=weights, reduction=reduction)

    def forward(self, pred, tar):
        # 将目标标签 one-hot 编码
        pred_one_hot = F.one_hot(pred, num_classes=pred.shape[1]).float()
        tar_one_hot = F.one_hot(tar, num_classes=pred.shape[1]).float()

        bce_loss_per_class = []

        for c in range(pred.shape[1]):
            utar = 1 - tar_one_hot[:,c,:,:]
            n_u = utar.sum() + self.eps
            n_c = tar_one_hot[:,c,:,:].sum() + self.eps
            loss = 0.5 * torch.sum(utar * torch.pow(pred_one_hot[:,c,:,:], 2)) / n_u + \
                   0.5 * torch.sum(tar_one_hot[:,c,:,:] * torch.pow(torch.clamp(self.m - pred_one_hot[:,c,:,:], min=0.0), 2)) / n_c
            bce_loss_per_class.append(loss)

        # 按照设定的reduction方式计算总损失
        if self.reduction == 'mean':
            return torch.stack(bce_loss_per_class).mean()
        elif self.reduction == 'sum':
            return torch.stack(bce_loss_per_class).sum()
        else:
            return bce_loss_per_class  # 返回未归一化的损失值