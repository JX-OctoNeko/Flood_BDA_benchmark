import os
import os.path as osp
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from skimage import io
from tqdm import tqdm

from sklearn.metrics import confusion_matrix

from core.trainer import Trainer
from utils.data_utils.misc import (
    to_array, to_pseudo_color,
    normalize_minmax, normalize_8bit,
    quantize_8bit as quantize,
)
from utils.utils import build_schedulers, HookHelper, FeatureContainer
from utils.metrics import (Meter, Precision, Recall, Accuracy, F1Score, kappa)


class CDTrainer(Trainer):
    def __init__(self, settings):
        super().__init__(settings['model'], settings['dataset'], settings['criterion'], settings['optimizer'], settings)

        # Set up tensorboard
        self.tb_on = (hasattr(self.logger, 'log_path') or self.debug) and self.ctx['tb_on']
        if self.tb_on:
            if hasattr(self.logger, 'log_path'):
                tb_dir = self.path(
                    'log',
                    osp.join('tb', osp.splitext(osp.basename(self.logger.log_path))[0], '.'),
                    name='tb',
                    auto_make=True,
                    suffix=False
                )
            else:
                tb_dir = self.path(
                    'log',
                    osp.join('tb', 'debug', '.'),
                    name='tb',
                    auto_make=True,
                    suffix=False
                )
                for root, dirs, files in os.walk(self.gpc.get_dir('tb'), False):
                    for f in files:
                        os.remove(osp.join(root, f))
                    for d in dirs:
                        os.rmdir(osp.join(root, d))
            self.tb_writer = SummaryWriter(tb_dir)
            self.logger.show_nl("TensorBoard logdir: {}\n".format(osp.abspath(self.gpc.get_dir('tb'))))
            self.tb_intvl = self.ctx['tb_intvl']

            # Global steps
            self.train_step = 0
            self.eval_step = 0

        # Whether to save network output
        self.save = self.ctx['save_on'] and not self.debug
        if self.save:
            self._mt_pool = ThreadPoolExecutor(max_workers=2)
        self.out_dir = self.ctx['out_dir']

        # Build lr schedulers
        self.sched_on = self.ctx['sched_on'] and self.is_training
        if self.sched_on:
            self.schedulers = build_schedulers(self.ctx['schedulers'], self.optimizer)

        self._init_trainer()

    def init_learning_rate(self):
        if not self.sched_on:
            self.lr = super().init_learning_rate()
        else:
            for idx, sched in enumerate(self.schedulers):
                if self.start_epoch > 0:
                    if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.logger.warn("The old state of lr scheduler {} will not be restored.".format(idx))
                        continue
                    # Restore previous state
                    # FIXME: This will trigger pytorch warning "Detected call of `lr_scheduler.step()` 
                    # before `optimizer.step()`" in pytorch 1.1.0 and later.
                    # Perhaps I should store the state of scheduler to a checkpoint file and restore it from disk.
                    last_epoch = self.start_epoch
                    while sched.last_epoch < last_epoch:
                        sched.step()
            self.lr = self.optimizer.param_groups[0]['lr']
        return self.lr

    def adjust_learning_rate(self, epoch, acc):
        if not self.sched_on:
            self.lr = super().adjust_learning_rate(epoch, acc)
        else:
            for sched in self.schedulers:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(acc)
                else:
                    sched.step()
            self.lr = self.optimizer.param_groups[0]['lr']
        return self.lr

    def compute_kl_divergence(self, p, q):
        kl = torch.sum(p * (torch.log(p + 1e-8) - torch.log(q + 1e-8)), dim=1)
        return torch.mean(kl)

    def train_epoch(self, epoch):
        losses = Meter()
        len_train = len(self.train_loader)
        width = len(str(len_train))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.train_loader)

        self.model.train()

        for i, (t1, t2, tar) in enumerate(pb):
            t1, t2, tar = self._prepare_data(t1, t2, tar)

            show_imgs_on_tb = self.tb_on and (i % self.tb_intvl == 0)

            fetch_dict = self._set_fetch_dict()
            out_dict = FeatureContainer()

            with HookHelper(self.model, fetch_dict, out_dict, hook_type='forward_out'):
                out = self.model(t1, t2)

            """
            修改loss: 将伪标签判断
            """
            w_l = []  # with label
            wo_l = []  # without label
            for p in range(tar.size(0)):
                # 使用eq检查灰度值
                if torch.any(torch.eq(tar[p], 6)):
                    # 如果像素值有6就添加索引到列表
                    wo_l.append(p)
                else:
                    w_l.append(p)

            # print(f"无标签数据：{wo_l}")
            # print(f"有标签数据：{w_l}")
            # 初始化自适应分布缓冲区
            buffer_size = 10  # 可以根据实际情况调整缓冲区大小
            adaptive_dist = AdaptiveDistribution(buffer_size)

            # print(len(out), type(out))
            # print(out.shape)
            # print(out[0].shape)
            # print(out[1].shape)

            pred = self._process_model_out(out)
            # print(type(pred))
            # print(len(pred))
            # print(pred[0].shape)
            # print(pred[1].shape)
            # if isinstance(pred, list):
            #     probs_pred = torch.zeros((8,5,256,256)).to(self.device)
            #     for pred_item in pred:
            #         print(pred_item.shape)
            #         probs_pred += self._pred_to_prob(pred_item)
            # else:
            #     probs_pred = self._pred_to_prob(pred) # 8 5 256 256, 概率图
            probs_pred = self._pred_to_prob(pred)
            # print(type(probs_pred))
            if isinstance(probs_pred, list):
                prob = []
                for pp in probs_pred:
                    _, prob_tmp = torch.max(pp, dim=1)  # 8 256 256 # 预测图
                    prob.append(prob_tmp)
            else:
                _, prob = torch.max(probs_pred, dim=1)  # 8 256 256 # 预测图

            # print(type(probs_pred))
            # print(probs_pred.shape)
            # print("out的shape是", out.shape)
            # print("pred的shape是", pred.shape)
            # print("probs_pred的shape是", probs_pred.shape)
            # print("prob的shape是", prob.shape)

            # print("out的tensor_0是", out[0][0])
            # print("out的tensor_1是", out[0][1])
            # print("out的tensor_2是", out[0][2])
            # print("out的tensor_3是", out[0][3])
            # print("out的tensor_4是", out[0][4])

            # print("pred的tensor是", pred)
            # print("probs_pred的tensor是", probs_pred)
            # print("prob的tensor是", prob)
            # print("out的tensor_1是", torch.any(torch.tensor(prob[0] == 1)))
            # print("out的tensor_2是", torch.any(torch.tensor(prob[0] == 2)))
            # print("out的tensor_3是", torch.any(torch.tensor(prob[0] == 3)))
            # print("out的tensor_4是", torch.any(torch.tensor(prob[0] == 4)))

            # print(len(pred))
            # print(type(pred))
            # print(pred[0].shape)
            # print(pred[1].shape)
            # print(len( probs_pred))
            # print(type( probs_pred))
            # print( probs_pred[0].shape)
            # print( probs_pred[1].shape)
            # print(pred.shape)
            # print(prob.shape)
            # print(probs_pred.shape)

            """
            判断without_label元组是否为空
            """

            if len(wo_l) == 0:
                # print("ok--1")
                loss = self.criterion(probs_pred, tar)
            elif len(w_l) == 0:
                # print("ok--2")
                # 获取有标签和无标签数据
                # labeled_images = tar[torch.tensor(w_l), :, :]  # 取出已知标签的样本
                # unlabeled_images = tar[torch.tensor(wo_l), :, :]  # 取出未知标签（标签为6）的样本

                labeled_images = tar[w_l, :, :]  # 取出已知标签的样本
                unlabeled_images = tar[wo_l, :, :]  # 取出未知标签（标签为6）的样本

                # supervised_loss = self.criterion(pred[torch.tensor(w_l), :, :], labeled_images)

                # supervised_loss = self.criterion(probs_pred[w_l, :, :], labeled_images)

                # 对于无标签数据，模型预测概率分布并计算熵最小化和KL散度损失
                # probs_unlabeled = probs_pred[torch.tensor(wo_l), :, :, :]

                """
                1.将fake label预测出来的prediction作为一致性正则化的干扰，计算kl loss
                """
                # if isinstance(probs_pred, list):
                #     # print(probs_pred[1].shape)
                #     probs_unlabeled = torch.zeros_like(probs_pred[0][wo_l, ...])
                #     # print(probs_unlabeled.shape)
                #     # print("ok")
                #     for l in probs_pred:
                #         # print(probs_unlabeled.shape)
                #         # print(l[wo_l, ...].shape)
                #         probs_unlabeled += l[wo_l, ...]
                # else:
                #     probs_unlabeled = probs_pred[wo_l, ...]
                #
                # entropy_loss = -torch.mean(torch.sum(probs_unlabeled * torch.log(probs_unlabeled + 1e-8), dim=(1,)))
                #
                # with torch.no_grad():
                #     adaptive_dist.update(probs_unlabeled.mean(dim=(2, 3)))  # 取平均概率并更新自适应分布
                # adaptive_reference_distribution = adaptive_dist.get_adaptive_distribution()
                # adaptive_reference_distribution = adaptive_reference_distribution.to(self.device)
                #
                # # print(probs_unlabeled.reshape(-1,5).shape)
                # # print(probs_unlabeled.reshape(-1,5).shape[0])
                # # print(adaptive_reference_distribution.shape)
                # # print(adaptive_reference_distribution.unsqueeze(0).shape)
                #
                # # 计算KL散度损失
                # kl_divergence_loss = self.compute_kl_divergence(probs_unlabeled.reshape(-1, 5),
                #                                            adaptive_reference_distribution.unsqueeze(0).repeat(
                #                                                probs_unlabeled.reshape(-1, 5).shape[0], 1))
                #
                # # 定义总无监督损失，并添加权重
                # # print(entropy_loss)
                # # print(kl_divergence_loss)
                # unsupervised_weight_entropy = 0.001
                # unsupervised_weight_kl = 0.001 # 根据lr设定权重，全监督loss的1/10
                # total_unsupervised_loss = unsupervised_weight_entropy * entropy_loss + \
                #                           unsupervised_weight_kl * kl_divergence_loss

                """
                *2.将true label预测出来的prediction作为一致性正则化的干扰，计算kl loss 这个在这里是不成立的，因为本来就没有标签，所以就没有分布
                """
                # if isinstance(probs_pred, list):
                #     probs_unlabeled = torch.zeros_like(probs_pred[0][wo_l, ...])
                #     for l in probs_pred:
                #         probs_unlabeled += l[wo_l, ...]
                # else:
                #     probs_unlabeled = probs_pred[wo_l, ...]
                #
                # entropy_loss = -torch.mean(torch.sum(probs_unlabeled * torch.log(probs_unlabeled + 1e-8), dim=(1,)))
                # total_unsupervised_loss = entropy_loss

                """
                3.将true label作为一致性正则化的干扰，计算kl loss
                """
                # if isinstance(probs_pred, list):
                #     probs_unlabeled = torch.zeros_like(probs_pred[0][wo_l, ...])
                #     for l in probs_pred:
                #         probs_unlabeled += l[wo_l, ...]
                # else:
                #     probs_unlabeled = probs_pred[wo_l, ...]
                #
                # entropy_loss = -torch.mean(torch.sum(probs_unlabeled * torch.log(probs_unlabeled + 1e-8), dim=(1,)))
                # total_unsupervised_loss = entropy_loss

                """
                4.将综合策略做组
                """
                if isinstance(probs_pred, list):
                    probs_unlabeled = torch.zeros_like(probs_pred[0][wo_l, ...])
                    for l in probs_pred:
                        probs_unlabeled += l[wo_l, ...]
                else:
                    probs_unlabeled = probs_pred[wo_l, ...]

                entropy_loss = -torch.mean(torch.sum(probs_unlabeled * torch.log(probs_unlabeled + 1e-8), dim=(1,)))

                with torch.no_grad():
                    adaptive_dist.update(probs_unlabeled.mean(dim=(2, 3)))  # 取平均概率并更新自适应分布
                adaptive_reference_distribution = adaptive_dist.get_adaptive_distribution()
                adaptive_reference_distribution = adaptive_reference_distribution.to(self.device)

                # 计算KL散度损失
                kl_divergence_loss = self.compute_kl_divergence(probs_unlabeled.reshape(-1, 5),
                                                                adaptive_reference_distribution.unsqueeze(0).repeat(
                                                                    probs_unlabeled.reshape(-1, 5).shape[0], 1))

                # 定义总无监督损失，并添加权重
                unsupervised_weight_entropy = 0.001
                unsupervised_weight_kl = 0.001  # 根据lr设定权重，全监督loss的1/10
                total_unsupervised_loss = unsupervised_weight_entropy * entropy_loss + \
                                          unsupervised_weight_kl * kl_divergence_loss

                """
                前面其实如果没有真实标签就采用熵最小化的方法优化，有真实标签才会使用到kl loss采用一致性归一化
                """
                # 计算总的损失并优化
                loss = total_unsupervised_loss
                # loss = 0 * entropy_loss # 计算benchmark使用的，为了是把loss变为0

            else:
                # print("ok--3")
                # 获取有标签和无标签数据
                # labeled_images = tar[torch.tensor(w_l), :, :]  # 取出已知标签的样本
                # unlabeled_images = tar[torch.tensor(wo_l), :, :]  # 取出未知标签（标签为6）的样本

                labeled_images = tar[w_l, :, :]  # 取出已知标签的样本
                unlabeled_images = tar[wo_l, :, :]  # 取出未知标签（标签为6）的样本

                # supervised_loss = self.criterion(pred[torch.tensor(w_l), :, :], labeled_images)

                if isinstance(probs_pred, list):
                    # supervised_loss = torch.tensor(0).to(self.device)
                    pp = []
                    for l in probs_pred:
                        # supervised_loss += self.criterion(l[w_l, :, :], labeled_images)
                        pp.append(l[w_l, ...])
                    supervised_loss = self.criterion(pp, labeled_images)
                else:
                    supervised_loss = self.criterion(probs_pred[w_l, :, :], labeled_images)

                # supervised_loss = self.criterion(probs_pred[w_l, :, :], labeled_images)

                """
                1.将fake label预测出来的prediction作为一致性正则化的干扰，计算kl loss
                """
                # # 对于无标签数据，模型预测概率分布并计算熵最小化和KL散度损失
                # # probs_unlabeled = probs_pred[torch.tensor(wo_l), :, :, :]
                #
                # if isinstance(probs_pred, list):
                #     # print(probs_pred[1].shape)
                #     probs_unlabeled = torch.zeros_like(probs_pred[0][wo_l, ...])
                #     # print(probs_unlabeled.shape)
                #     # print("ok")
                #     for l in probs_pred:
                #         # print(probs_unlabeled.shape)
                #         # print(l[wo_l, ...].shape)
                #         probs_unlabeled += l[wo_l, ...]
                # else:
                #     probs_unlabeled = probs_pred[wo_l, ...]
                #
                #
                # entropy_loss = -torch.mean(torch.sum(probs_unlabeled * torch.log(probs_unlabeled + 1e-8), dim=(1,)))
                #
                # # 更新自适应分布并获取参考分布
                #
                # with torch.no_grad():
                #     adaptive_dist.update(probs_unlabeled.mean(dim=(2, 3)))  # 取平均概率并更新自适应分布
                # adaptive_reference_distribution = adaptive_dist.get_adaptive_distribution()
                # adaptive_reference_distribution = adaptive_reference_distribution.to(self.device)
                #
                # # print(probs_unlabeled.reshape(-1,5).shape)
                # # print(probs_unlabeled.reshape(-1,5).shape[0])
                # # print(adaptive_reference_distribution.shape)
                # # print(adaptive_reference_distribution.unsqueeze(0).shape)
                #
                # # 计算KL散度损失
                # kl_divergence_loss = self.compute_kl_divergence(probs_unlabeled.reshape(-1, 5),
                #                                            adaptive_reference_distribution.unsqueeze(0).repeat(
                #                                                probs_unlabeled.reshape(-1, 5).shape[0], 1))
                #
                # # 定义总无监督损失，并添加权重
                # unsupervised_weight_entropy = 0.001
                # unsupervised_weight_kl = 0.001 # 根据lr设定权重，全监督loss的1/10
                # total_unsupervised_loss = unsupervised_weight_entropy * entropy_loss + \
                #                           unsupervised_weight_kl * kl_divergence_loss

                """
                2.将true label预测出来的prediction作为一致性正则化的干扰，计算kl loss
                """
                # # 对于无标签数据，模型预测概率分布并计算熵最小化和KL散度损失
                # if isinstance(probs_pred, list):
                #     probs_labeled = torch.zeros_like(probs_pred[0][w_l, ...])
                #     probs_unlabeled = torch.zeros_like(probs_pred[0][w_l, ...])
                #     for l in probs_pred:
                #         probs_labeled += l[w_l, ...]
                #         probs_unlabeled += l[wo_l, ...]
                # else:
                #     probs_labeled = probs_pred[w_l, ...]
                #     probs_unlabeled = probs_pred[wo_l, ...]
                #
                # entropy_loss = -torch.mean(torch.sum(probs_unlabeled * torch.log(probs_unlabeled + 1e-8), dim=(1,)))
                #
                # # 更新自适应分布并获取参考分布
                #
                # with torch.no_grad():
                #     adaptive_dist.update(probs_labeled.mean(dim=(2, 3)))  # 取平均概率并更新自适应分布
                # adaptive_reference_distribution = adaptive_dist.get_adaptive_distribution()
                # adaptive_reference_distribution = adaptive_reference_distribution.to(self.device)
                #
                # # 计算KL散度损失
                # kl_divergence_loss = self.compute_kl_divergence(probs_unlabeled.reshape(-1, 5),
                #                                                 adaptive_reference_distribution.unsqueeze(0).repeat(
                #                                                     probs_unlabeled.reshape(-1, 5).shape[0], 1))
                #
                # # 定义总无监督损失，并添加权重
                # unsupervised_weight_entropy = 0.001
                # unsupervised_weight_kl = 0.001  # 根据lr设定权重，全监督loss的1/10
                # total_unsupervised_loss = unsupervised_weight_entropy * entropy_loss + \
                #                           unsupervised_weight_kl * kl_divergence_loss

                """
                3.将true label作为一致性正则化的干扰，计算kl loss
                """
                # # 对于无标签数据，模型预测概率分布并计算熵最小化和KL散度损失
                # if isinstance(probs_pred, list):
                #     probs_unlabeled = torch.zeros_like(probs_pred[0][w_l, ...])
                #     for l in probs_pred:
                #         probs_unlabeled += l[wo_l, ...]
                # else:
                #     probs_unlabeled = probs_pred[wo_l, ...]
                #
                # entropy_loss = -torch.mean(torch.sum(probs_unlabeled * torch.log(probs_unlabeled + 1e-8), dim=(1,)))
                #
                # # 更新自适应分布并获取参考分布
                # true_labeled_group = tar[w_l, ...]
                #
                # # 首先将张量展平为形状为[x, 256*256]的张量
                # flat_true_labeled_group = true_labeled_group.view(true_labeled_group.size(0), -1)
                #
                # # 统计每个类别在所有图片中的像素数量
                # class_counts = torch.zeros(5)  # 创建一个形状为[5]的全零张量，用于统计每个类别的像素数量
                # for i in range(5):
                #     class_counts[i] = (flat_true_labeled_group == i).sum(dim=1).sum()  # 统计每个类别的像素数量
                #
                # # 计算每个类别在所有图片中的比例
                # total_pixels = flat_true_labeled_group.size(1) * flat_true_labeled_group.size(0)  # 计算总像素数量
                # adaptive_reference_distribution = class_counts / total_pixels  # 计算每个类别的比例
                #
                # adaptive_reference_distribution = adaptive_reference_distribution.to(self.device)
                #
                # # 计算KL散度损失
                # kl_divergence_loss = self.compute_kl_divergence(probs_unlabeled.reshape(-1, 5),
                #                                                 adaptive_reference_distribution.unsqueeze(0).repeat(
                #                                                     probs_unlabeled.reshape(-1, 5).shape[0], 1))
                #
                # # 定义总无监督损失，并添加权重
                # unsupervised_weight_entropy = 0.001
                # unsupervised_weight_kl = 0.001  # 根据lr设定权重，全监督loss的1/10
                # total_unsupervised_loss = unsupervised_weight_entropy * entropy_loss + \
                #                           unsupervised_weight_kl * kl_divergence_loss

                """
                4.将3种合一作为一致性正则化的干扰，计算kl loss
                """
                # 对于无标签数据，模型预测概率分布并计算熵最小化和KL散度损失
                if isinstance(probs_pred, list):
                    probs_labeled = torch.zeros_like(probs_pred[0][w_l, ...])
                    probs_unlabeled = torch.zeros_like(probs_pred[0][w_l, ...])
                    for l in probs_pred:
                        probs_labeled += l[w_l, ...]
                        probs_unlabeled += l[wo_l, ...]
                else:
                    probs_labeled = probs_pred[w_l, ...]
                    probs_unlabeled = probs_pred[wo_l, ...]

                entropy_loss = -torch.mean(torch.sum(probs_unlabeled * torch.log(probs_unlabeled + 1e-8), dim=(1,)))

                # 更新自适应分布并获取参考分布
                """
                第一种策略，使用伪标签预测值做组获得的kl loss
                """
                with torch.no_grad():
                    adaptive_dist.update(probs_unlabeled.mean(dim=(2, 3)))  # 取平均概率并更新自适应分布
                adaptive_reference_distribution_1 = adaptive_dist.get_adaptive_distribution()
                adaptive_reference_distribution_1 = adaptive_reference_distribution_1.to(self.device)

                # 计算KL散度损失
                kl_divergence_loss_1 = self.compute_kl_divergence(probs_unlabeled.reshape(-1, 5),
                                                                  adaptive_reference_distribution_1.unsqueeze(0).repeat(
                                                                      probs_unlabeled.reshape(-1, 5).shape[0], 1))
                """
                第二种策略，使用真实标签预测值做组获得的kl loss
                """
                with torch.no_grad():
                    adaptive_dist.update(probs_labeled.mean(dim=(2, 3)))  # 取平均概率并更新自适应分布
                adaptive_reference_distribution_2 = adaptive_dist.get_adaptive_distribution()
                adaptive_reference_distribution_2 = adaptive_reference_distribution_2.to(self.device)

                # 计算KL散度损失
                kl_divergence_loss_2 = self.compute_kl_divergence(probs_unlabeled.reshape(-1, 5),
                                                                  adaptive_reference_distribution_2.unsqueeze(0).repeat(
                                                                      probs_unlabeled.reshape(-1, 5).shape[0], 1))

                """
                第三种策略，使用真实标签做组获得的kl loss
                """
                # 更新自适应分布并获取参考分布
                true_labeled_group = tar[w_l, ...]

                # 首先将张量展平为形状为[x, 256*256]的张量
                flat_true_labeled_group = true_labeled_group.view(true_labeled_group.size(0), -1)

                # 统计每个类别在所有图片中的像素数量
                class_counts = torch.zeros(5)  # 创建一个形状为[5]的全零张量，用于统计每个类别的像素数量
                for i in range(5):
                    class_counts[i] = (flat_true_labeled_group == i).sum(dim=1).sum()  # 统计每个类别的像素数量

                # 计算每个类别在所有图片中的比例
                total_pixels = flat_true_labeled_group.size(1) * flat_true_labeled_group.size(0)  # 计算总像素数量
                adaptive_reference_distribution_3 = class_counts / total_pixels  # 计算每个类别的比例

                adaptive_reference_distribution_3 = adaptive_reference_distribution_3.to(self.device)

                # 计算KL散度损失
                kl_divergence_loss_3 = self.compute_kl_divergence(probs_unlabeled.reshape(-1, 5),
                                                                  adaptive_reference_distribution_3.unsqueeze(0).repeat(
                                                                      probs_unlabeled.reshape(-1, 5).shape[0], 1))

                # 定义总无监督损失，并添加权重
                unsupervised_weight_entropy = 0.001
                unsupervised_weight_kl_1 = 0.001  # 根据lr设定权重，全监督loss的1/10
                unsupervised_weight_kl_2 = 0.001  # 根据lr设定权重，全监督loss的1/10
                unsupervised_weight_kl_3 = 0.001  # 根据lr设定权重，全监督loss的1/10
                total_unsupervised_loss = unsupervised_weight_entropy * entropy_loss + \
                                          unsupervised_weight_kl_1 * kl_divergence_loss_1 + \
                                          unsupervised_weight_kl_2 * kl_divergence_loss_2 + \
                                          unsupervised_weight_kl_3 * kl_divergence_loss_3

                """
                前面的注释是3种策略产生的unsupervised loss
                """
                # 计算总的损失并优化
                loss = total_unsupervised_loss + supervised_loss
                # loss = supervised_loss
                # print(supervised_loss)
                # print(total_unsupervised_loss)

            # 1.14 课上修改
            # loss = 0
            # for p in pred:
            #     loss += self.criterion(pred, tar)

            # 原始代码
            # loss = self.criterion(pred, tar)
            losses.update(loss.item(), n=tar.shape[0])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            desc = (start_pattern + " Loss: {:.4f} ({:.4f})").format(i + 1, len_train, losses.val, losses.avg)

            pb.set_description(desc)
            if i % max(1, len_train // 10) == 0:
                self.logger.dump(desc)

            if self.tb_on:
                # Write to tensorboard
                self.tb_writer.add_scalar("Train/running_loss", losses.val, self.train_step)
                if show_imgs_on_tb:
                    t1, t2 = to_array(t1[0]), to_array(t2[0])
                    t1, t2 = self._denorm_image(t1), self._denorm_image(t2)
                    t1, t2 = self._process_input_pairs(t1, t2)
                    self.tb_writer.add_image("Train/t1_picked", t1, self.train_step, dataformats='HWC')
                    self.tb_writer.add_image("Train/t2_picked", t2, self.train_step, dataformats='HWC')
                    self.tb_writer.add_image("Train/labels_picked", to_array(tar[0]), self.train_step, dataformats='HW')
                    for key, feats in out_dict.items():
                        for idx, feat in enumerate(feats):
                            feat = self._process_fetched_feat(feat)
                            self.tb_writer.add_image(f"Train/{key}_{idx}", feat, self.train_step, dataformats='HWC')
                    self.tb_writer.flush()
                self.train_step += 1

        if self.tb_on:
            self.tb_writer.add_scalar("Train/loss", losses.avg, self.train_step)
            self.tb_writer.add_scalar("Train/lr", self.lr, self.train_step)

    def evaluate_epoch(self, epoch):
        self.logger.show_nl("Epoch: [{0}]".format(epoch))
        losses = Meter()
        len_eval = len(self.eval_loader)
        width = len(str(len_eval))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.eval_loader)

        # Construct metrics
        metrics = (Precision(mode='accum'), Recall(mode='accum'), F1Score(mode='accum'), Accuracy(mode='accum'),
                   kappa(mode="accum"))
        # new_metrics = (Precision(mode='accum'), Recall(mode='accum'), F1Score(mode='accum'), Accuracy(mode='accum'))

        self.model.eval()
        # num_classes = 5
        # precision_per_class = np.zeros(num_classes)
        # recall_per_class = np.zeros(num_classes)
        # f1_per_class = np.zeros(num_classes)

        with torch.no_grad():
            total_cm = numpy.zeros((2, 2))  # 2x2 cm
            plt_cm = numpy.zeros((4, 4))  # 画混淆矩阵用

            for i, (name, t1, t2, tar) in enumerate(pb):
                t1, t2, tar = self._prepare_data(t1, t2, tar)
                batch_size = tar.shape[0]

                fetch_dict = self._set_fetch_dict()
                out_dict = FeatureContainer()

                with HookHelper(self.model, fetch_dict, out_dict, hook_type='forward_out'):
                    out = self.model(t1, t2)

                """
                修改loss: 将伪标签判断
                """
                w_l = []  # with label
                for p in range(tar.size(0)):
                    # 使用eq检查灰度值
                    if torch.any(torch.eq(tar[p], 6)):
                        pass
                    else:
                        w_l.append(p)

                pred = self._process_model_out(out)
                probs_pred = self._pred_to_prob(pred)  # 8 5 256 256, 概率图
                # print(type(probs_pred))
                if isinstance(probs_pred, list):
                    prob = []
                    prob_top2 = []
                    for pp in probs_pred:
                        _, prob_tmp = torch.max(pp, dim=1)  # 8 256 256 # 预测图
                        _, prob_tmp_top2 = pp.topk(k=2, dim=1)  # 8 256 256 # 预测图
                        prob_tmp_top2 = prob_tmp_top2[:, 1, :]
                        prob.append(prob_tmp)
                        prob_top2.append(prob_tmp_top2)
                else:
                    _, prob = torch.max(probs_pred, dim=1)  # 8 256 256 # 预测图
                    _, prob_top2 = probs_pred.topk(k=2, dim=1)
                    prob_top2 = prob_top2[:, 1, :]

                # _, prob = torch.max(probs_pred, dim=1)  # 8 256 256 # 预测图

                # print(pred[w_l, ...].shape)
                # print(pred[wo_l, ...].shape)
                # print(pred.shape)
                # print(tar.shape)

                """
                判断without_label元组是否为空
                """
                if len(w_l) == 0:
                    # loss = torch.tensor(0)
                    losses.update(0, n=1)
                else:
                    # 获取有标签和无标签数据
                    # labeled_images = tar[torch.tensor(w_l), :, :]  # 取出已知标签的样本
                    labeled_images = tar[w_l, :, :]  # 取出已知标签的样本

                    # loss = self.criterion(pred[torch.tensor(w_l), :, :], labeled_images)
                    # loss = self.criterion(probs_pred[w_l, :, :], labeled_images)

                    if isinstance(probs_pred, list):
                        pp = []
                        for l in probs_pred:
                            pp.append(l[w_l, ...])
                        loss = self.criterion(pp, labeled_images)
                    else:
                        loss = self.criterion(probs_pred[w_l, :, :], labeled_images)

                    losses.update(loss.item(), n=len(w_l))

                labeled_images = labeled_images

                # Convert to numpy arrays
                # pred = pred[torch.tensor(w_l), :, :].cpu().numpy()

                # prob = prob[w_l, :, :].cpu().numpy()
                if isinstance(prob, list):
                    prob = prob[0][w_l, ...].cpu().numpy()
                    prob_top2 = prob_top2[0][w_l, ...].cpu().numpy()
                else:
                    prob = prob[w_l, :, :].cpu().numpy()
                    prob_top2 = prob_top2[w_l, ...].cpu().numpy()

                # print(prob.shape)
                # desired_value = 4 # 选择要检查的值
                # contains_desired_value = np.any(prob == desired_value)
                # 打印结果
                # print(f"tensor中是否含有像素值{desired_value}:{contains_desired_value}")
                # cm = (prob>0.5).astype('uint8')

                cm = prob.astype('uint8')

                # if (cm == 3).any():
                #     print("有3像素")
                # if (cm == 4).any():
                #     print("有4像素")

                cm_top2 = prob_top2.astype('uint8')
                # print(cm.shape, cm_top2.shape)
                mask = (cm == 0)
                cm[mask] = cm_top2[mask]
                # print(cm[0])
                # print(tar[0])

                # print(cm.shape)
                # tar = tar[torch.tensor(w_l), :, :].cpu().numpy().astype('uint8')
                tar = tar[w_l, :, :].cpu().numpy().astype('uint8')
                mask = (tar == 0)
                cm[mask] = tar[mask]
                # print(cm[0])
                # print(tar[0])

                for m in metrics:
                    m.update(cm, tar, n=len(w_l))

                # 混淆矩阵4分类
                one_plt_cm = confusion_matrix((tar[tar != 0] - 1).ravel(), (cm[tar != 0] - 1).ravel())
                if plt_cm.shape == one_plt_cm.shape:
                    if isinstance(one_plt_cm, list):
                        plt_cm = sum(val for val in one_plt_cm)
                    else:
                        new_n = len(w_l)
                        plt_cm += one_plt_cm * new_n

                # if (cm == 3).any():
                #     print("有3像素")
                # if (cm == 4).any():
                #     print("有4像素")
                # 这里cm变量还有3，4值

                # 在这里就是能计算2分类结果
                new_cm = cm
                new_mask = (new_cm > 1)
                new_cm[new_mask] = 2
                new_tar_mask = (tar > 1)
                new_tar = tar
                new_tar[new_tar_mask] = 2

                # # 这之后就没有了3，4
                # # if (cm == 3).any():
                # #     print("有1像素")
                # # if (cm == 4).any():
                # #     print("有2像素")

                batch_cm = confusion_matrix((new_tar[tar != 0] - 1).ravel(), (new_cm[tar != 0] - 1).ravel())
                # # print(batch_cm)
                if total_cm.shape == batch_cm.shape:
                    total_cm += batch_cm
                    # print(total_cm.shape) # 2*2
                #     # print("calculate + 1 !")
                #     # print(total_cm.shape)
                #     # print(total_cm)

                #
                # for new_m in new_metrics:
                #
                #     new_m.update(new_cm, new_tar, n=len(w_l))

                # for j in range(num_classes):
                #     tp = m._cm.val[j, j]
                #     fp = np.sum(m._cm.val[:, j]) - tp
                #     fn = np.sum(m._cm.val[j, :]) - tp
                #
                #     # Precision
                #     precision_per_class[j] += tp / max(tp + fp, 1e-9)
                #
                #     # Recall
                #     recall_per_class[j] += tp / max(tp + fn, 1e-9)
                #
                #     # F1 Score
                #     f1_per_class[j] += 2 * (precision_per_class[j] * recall_per_class[j]) / max(
                #         precision_per_class[j] + recall_per_class[j], 1e-9)

                desc = (start_pattern + " Loss: {:.4f} ({:.4f})").format(i + 1, len_eval, losses.val, losses.avg)
                for m in metrics:
                    if isinstance(m.sum, tuple):
                        ss = ["{:.4f}".format(val) for val in m.val]
                    else:
                        ss = ["{}".format(m.val)]
                    desc += "{} {}".format(m.__name__, "___".join(ss))

                # new_desc = (start_pattern + " Loss: {:.4f} ({:.4f})").format(i + 1, len_eval, losses.val, losses.avg)
                # for new_m in new_metrics:
                #     if isinstance(new_m.val, tuple):
                #         new_ss = ["{:.4f}".format(val) for val in new_m.val]
                #     else:
                #         new_ss = ["{}".format(new_m.val)]
                #     desc += "{} {}".format(new_m.__name__, "___".join(new_ss))

                # desc = (start_pattern+" Loss: {:.4f} ({:.4f})").format(i+1, len_eval, losses.val, losses.avg)
                # for m in metrics:
                #     desc += " {} {:.4f}".format(m.__name__, m.val)

                pb.set_description(desc)

                # pb.set_description((new_desc))
                dump = not self.is_training or (i % max(1, len_eval // 10) == 0)
                if dump:
                    self.logger.dump(desc)
                    # self.logger.dump(new_desc)

                if self.tb_on:
                    if dump:
                        for j in range(batch_size):
                            t1_, t2_ = to_array(t1[j]), to_array(t2[j])
                            t1_, t2_ = self._denorm_image(t1_), self._denorm_image(t2_)
                            t1_, t2_ = self._process_input_pairs(t1_, t2_)
                            self.tb_writer.add_image("Eval/t1", t1_, self.eval_step, dataformats='HWC')
                            self.tb_writer.add_image("Eval/t2", t2_, self.eval_step, dataformats='HWC')
                            self.tb_writer.add_image("Eval/labels", quantize(tar[j]), self.eval_step, dataformats='HW')
                            self.tb_writer.add_image("Eval/prob", to_pseudo_color(quantize(prob[j])), self.eval_step,
                                                     dataformats='HWC')
                            self.tb_writer.add_image("Eval/cm", quantize(cm[j]), self.eval_step, dataformats='HW')
                            for key, feats in out_dict.items():
                                for idx, feat in enumerate(feats):
                                    feat = self._process_fetched_feat(feat[j])
                                    self.tb_writer.add_image(f"Eval/{key}_{idx}", feat, self.eval_step,
                                                             dataformats='HWC')
                            self.eval_step += 1
                    else:
                        self.eval_step += batch_size

                # if (cm == 3).any():
                #     print("有3像素")
                # if (cm == 4).any():
                #     print("有4像素")

                if self.save:
                    # cm_j_1_num = 0
                    # cm_j_2_num = 0
                    # cm_j_3_num = 0
                    # cm_j_4_num = 0
                    for j in range(batch_size):
                        # print(np.unique(cm[j])) # 去除重复值
                        # if (cm[j] == 1).any():
                        #     cm_j_1_num += 1
                        # elif (cm[j] == 2).any():
                        #     cm_j_2_num += 1
                        # elif (cm[j] == 3).any():
                        #     cm_j_3_num += 1
                        # elif (cm[j] == 4).any():
                        #     cm_j_4_num += 1
                        # print(cm_j_1_num)
                        # print(cm_j_2_num)
                        # print(cm_j_3_num)
                        # print(cm_j_4_num)

                        # if (cm == 3).any():
                        #     print("有3像素")
                        # if (cm == 4).any():
                        #     print("有4像素")

                        self.save_image(name[j], quantize(cm[j]), epoch)

            # specificities = []
            # sensitivities = []
            # for i in range(4):
            #     tn, fp, fn, tp = total_cm[i, :].ravel()
            #     specificity = tn / (tn + fp) if tn + fp > 0 else np.nan  # 防止分母为0
            #     sensitivity = tp / (tp + fn) if tp + fn > 0 else np.nan  # 防止分母为0
            #     specificities.append(specificity)
            #     sensitivities.append(sensitivity)

            # print("specific:", specificities)
            # print("sensitivity", sensitivities)

            # 建立计算macro的函数
            def calculate_macro_acc(confusion_matrix):
                num_classes = len(confusion_matrix)
                accuracies = []

                for i in range(num_classes):
                    tp = confusion_matrix[i][i]  # True positive
                    fp = np.sum(confusion_matrix[:, i]) - tp  # False positive
                    fn = np.sum(confusion_matrix[i, :]) - tp  # False negative
                    tn = np.sum(confusion_matrix) - tp - fp - fn  # True negative

                    if (tp + fp + fn + tn) == 0:
                        accuracy = 0
                    else:
                        accuracy = tp / (tp + fp)

                    accuracies.append(accuracy)

                macro_accuracy = np.mean(accuracies)
                return macro_accuracy

            # # 为了可以产生4分类的save
            # new_cm = cm
            # new_mask = (new_cm > 1)
            # new_cm[new_mask] = 2
            # new_tar_mask = (tar > 1)
            # new_tar = tar
            # new_tar[new_tar_mask] = 2
            #
            # batch_cm = confusion_matrix((new_tar[tar != 0] - 1).ravel(), (new_cm[tar != 0] - 1).ravel())
            # # print(batch_cm)
            # if total_cm.shape == batch_cm.shape:
            #     total_cm += batch_cm

            def calculate_Kappa(cm):
                p_o = np.nan_to_num(np.diag(cm).sum() / cm.sum())

                pe_rows = cm.sum(axis=0)
                pe_cols = cm.sum(axis=1)
                sum_total = sum(pe_cols)
                p_e = np.nan_to_num(np.dot(pe_rows, pe_cols) / float(sum_total ** 2))

                kappa_index = (p_o - p_e) / (1 - p_e)

                return kappa_index

            """
            绘制confusion matrix
            """

            # def plt_metric(cm):
            #
            #     import seaborn as sns
            #     import matplotlib.pyplot as plt
            #     labels = ["No Dmg.", "Minor Dmg.", "Major Dmg.", "Destroyed"]
            #     # fig, ax = plt.subplots()
            #     sns.heatmap(cm, annot=True, cmap="Blues", fmt='.3f', xticklabels=labels, yticklabels=labels,
            #                 annot_kws={"size": 20})
            #     # ax.set_title("Normalized Confusion Matrix")
            #     # ax.set_xlabel("Predicted Labels")
            #     # ax.set_ylabel("True Labels")
            #     cax = plt.gcf().axes[-1]
            #     cax.tick_params(labelsize=16)
            #
            #     plt.xticks(rotation=45, fontsize=20)
            #     plt.yticks(rotation=0, fontsize=20)
            #     outpath = r"C:\Users\student\Desktop\归一化混淆矩阵图"
            #     plt.savefig(os.path.join(outpath, "normalized_confusion_matrix_lunet.png"), bbox_inches="tight")
            #
            #     # plt.show()
            #
            # plt_cm_percent = plt_cm / plt_cm.sum(axis=1)[:, np.newaxis]
            # plt_metric(plt_cm_percent)

            precision = np.nan_to_num(np.diag(total_cm) / total_cm.sum(axis=0))
            print("precision:", precision)
            recall = np.nan_to_num(np.diag(total_cm) / total_cm.sum(axis=1))
            print("recall:", recall)
            f1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))
            print("f1-score:", f1)
            # acc = calculate_macro_acc(total_cm) # 计算macro的
            acc = np.nan_to_num(np.diag(total_cm).sum() / total_cm.sum())  # 计算micro的
            print("accuracy:", acc)
            kappa_index = calculate_Kappa(total_cm)
            print("Kappa-2class:", kappa_index)

            precision_4class = np.nan_to_num(np.diag(plt_cm) / plt_cm.sum(axis=0))
            print("precision-4class:", precision_4class)
            recall_4class = np.nan_to_num(np.diag(plt_cm) / plt_cm.sum(axis=1))
            print("recall-4class:", recall_4class)
            f1_4class = np.nan_to_num(2 * (precision_4class * recall_4class) / (precision_4class + recall_4class))
            print("f1-score-4class:", f1_4class)
            # acc = calculate_macro_acc(total_cm) # 计算macro的
            acc_4class = np.nan_to_num(np.diag(plt_cm).sum() / plt_cm.sum())  # 计算micro的
            print("accuracy-4class:", acc_4class)
            kappa_index_4class = calculate_Kappa(plt_cm)
            print("Kappa-4class:", kappa_index_4class)

        if self.tb_on:
            self.tb_writer.add_scalar("Eval/loss", losses.avg, self.eval_step)
            for m in metrics:
                self.tb_writer.add_scalar(f"Eval/{m.__name__.lower()}", m.val, self.eval_step)
            self.tb_writer.flush()

        return metrics[2].val  # F1-score

    def save_image(self, file_name, image, epoch):
        file_path = osp.join(
            'epoch_{}'.format(epoch),
            self.out_dir,
            file_name
        )
        out_path = self.path(
            'out', file_path,
            suffix=not self.ctx['suffix_off'],
            auto_make=True,
            underline=True
        )
        return self._mt_pool.submit(partial(io.imsave, check_contrast=False), out_path, image)  # 没问题 *3

    def _denorm_image(self, x):
        return x * np.asarray(self.ctx['sigma']) + np.asarray(self.ctx['mu'])

    def _process_input_pairs(self, t1, t2):
        vis_band_inds = self.ctx['tb_vis_bands']
        t1 = t1[..., vis_band_inds]
        t2 = t2[..., vis_band_inds]
        if self.ctx['tb_vis_norm'] == '8bit':
            t1 = normalize_8bit(t1)
            t2 = normalize_8bit(t2)
        else:
            t1 = normalize_minmax(t1)
            t2 = normalize_minmax(t2)
        t1 = np.clip(t1, 0.0, 1.0)
        t2 = np.clip(t2, 0.0, 1.0)
        return t1, t2

    def _process_fetched_feat(self, feat):
        feat = normalize_minmax(feat.mean(0))
        feat = quantize(to_array(feat))
        feat = to_pseudo_color(feat)
        return feat

    def _init_trainer(self):
        pass

    def _prepare_data(self, t1, t2, tar):
        return t1.to(self.device), t2.to(self.device), tar.to(self.device)

    def _set_fetch_dict(self):
        return dict()

    def _process_model_out(self, out):
        return out

    def _pred_to_prob(self, pred):
        return torch.nn.functional.softmax(pred, dim=1)
        # print("输入的pred是", type(pred))
        # return [torch.nn.functional.softmax(o, dim=1).squeeze(1) for o in pred]


# AdaptiveDistribution 类定义
class AdaptiveDistribution:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = torch.zeros(buffer_size, 5)
        self.index = 0

    def update(self, current_predictions):
        self.buffer[self.index] = current_predictions.detach().mean(dim=0)
        # self.index = (self.index + 1) % self.buffer_size
        self.index = (self.index + 1) % self.buffer_size

    def get_adaptive_distribution(self):
        # adaptive_distribution = self.buffer.mean(dim=0)
        adaptive_distribution = self.buffer.mean(dim=0)
        return adaptive_distribution / adaptive_distribution.sum(dim=0, keepdim=True)
