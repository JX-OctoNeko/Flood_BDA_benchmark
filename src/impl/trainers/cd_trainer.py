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
from utils.metrics import (Meter, Precision, Recall, Accuracy, F1Score, Kappa)


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
            self.strategy = settings['semi_trategy']

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

    # kl_divergence
    def compute_kl_divergence(self, p, q):
        kl = torch.sum(p * (torch.log(p + 1e-8) - torch.log(q + 1e-8)), dim=1)
        return torch.mean(kl)
    
    # Entropy minimization loss 
    def compute_entropy_loss(self, probs_unlabeled):
        # unsupervised entropy loss weight
        unsupervised_weight_entropy = 0.001

        # Entropy minimization
        entropy_loss = -torch.mean(torch.sum(probs_unlabeled * torch.log(probs_unlabeled + 1e-8), dim=(1,)))

        entropy_loss = unsupervised_weight_entropy * entropy_loss

        return entropy_loss


    def compute_kl_loss_strategy_1(self, probs_unlabeled, adaptive_dist):
        """
        Strategy1: Calculate the unsupervised loss using pseudo-labeled predictions as consistency regularization
        Calculate KL divergence
        """
        
        with torch.no_grad():
            adaptive_dist.update(probs_unlabeled.mean(dim=(2, 3)))  # Take the average probability and update the adaptive distribution
        adaptive_reference_distribution = adaptive_dist.get_adaptive_distribution()
        adaptive_reference_distribution = adaptive_reference_distribution.to(self.device)
        
        # Calculating KL loss
        kl_divergence_loss = self.compute_kl_divergence(probs_unlabeled.reshape(-1, 5),
                                                    adaptive_reference_distribution.unsqueeze(0).repeat(
                                                        probs_unlabeled.reshape(-1, 5).shape[0], 1))

        # Defining total unsupervised loss and add weight
        unsupervised_weight_kl = 0.001 # Setting weight based on lr, 1/10 of supervised loss
        kl_divergence_loss = unsupervised_weight_kl * kl_divergence_loss
        
        return kl_divergence_loss
    

    def compute_kl_loss_strategy_2(self, probs_labeled, probs_unlabeled, adaptive_dist):
        """
        Strategy2: Calculate the unsupervised loss using gournd-truth label predictions as consistency regularization
        Calculate KL divergence
        The difference from 1 is that the gournd-truth label predictions are used in the calculation using adaptive dist
        """
        
        # Updating adaptive distribution and get reference distribution
        with torch.no_grad():
            adaptive_dist.update(probs_labeled.mean(dim=(2, 3)))  # Getting mean probs and update adaptive distribution
        adaptive_reference_distribution = adaptive_dist.get_adaptive_distribution()
        adaptive_reference_distribution = adaptive_reference_distribution.to(self.device)
        
        # Calculating KL loss
        kl_divergence_loss = self.compute_kl_divergence(probs_unlabeled.reshape(-1, 5),
                                                        adaptive_reference_distribution.unsqueeze(0).repeat(
                                                            probs_unlabeled.reshape(-1, 5).shape[0], 1))
        
        # Defining total unsupervised loss and add weight
        unsupervised_weight_kl = 0.001  # Setting weight based on lr, 1/10 of supervised loss
        kl_divergence_loss = unsupervised_weight_kl * kl_divergence_loss

        return kl_divergence_loss
    

    def compute_kl_loss_strategy_3(self, tar, w_l, probs_unlabeled):
        """
        Strategy3: Calculate the unsupervised loss using gournd-truth label as consistency regularization
        Calculate KL divergence
        Instead of applying to the get_adaptive_distribution() class to obtain the reference distribution, the manually calculated
        """

        # Updating adaptive distribution and get reference distribution
        true_labeled_group = tar[w_l, ...]
        
        # First the tensor is flattened to a tensor of shape [x, 256*256]
        flat_true_labeled_group = true_labeled_group.view(true_labeled_group.size(0), -1)
        
        # Count the number of pixels in all images for each category
        class_counts = torch.zeros(5)  # Create an zero tensor of shape [5] for counting the number of pixels in each category
        for i in range(5):
            class_counts[i] = (flat_true_labeled_group == i).sum(dim=1).sum()  # Count the number of pixels in each category
        
        # Calculate the percentage of each category in all images
        total_pixels = flat_true_labeled_group.size(1) * flat_true_labeled_group.size(0)  # Calculate the total number of pixels
        adaptive_reference_distribution = class_counts / total_pixels  # Calculate the proportion of each category
        
        adaptive_reference_distribution = adaptive_reference_distribution.to(self.device)
        
        # Calculating KL loss
        kl_divergence_loss = self.compute_kl_divergence(probs_unlabeled.reshape(-1, 5),
                                                        adaptive_reference_distribution.unsqueeze(0).repeat(
                                                            probs_unlabeled.reshape(-1, 5).shape[0], 1))

        # Defining total unsupervised loss and add weight
        unsupervised_weight_kl = 0.001  # Setting weight based on lr, 1/10 of supervised loss
        kl_divergence_loss = unsupervised_weight_kl * kl_divergence_loss

        return kl_divergence_loss

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
            Include pseudo-labeling in calculations
            """
            w_l = []  # with label
            wo_l = []  # without label
            for p in range(tar.size(0)):
                # checking the gray value by eq
                if torch.any(torch.eq(tar[p], 6)):
                    # if value=6 add to list
                    wo_l.append(p)
                else:
                    w_l.append(p)

            # Initialize the adaptive distribution buffer
            buffer_size = 10  # Buffer size can be adjusted according to the actual situation
            adaptive_dist = AdaptiveDistribution(buffer_size)

            pred = self._process_model_out(out)

            probs_pred = self._pred_to_prob(pred)
            
            # Obtain a list of true and pseudo labels, in this experiment the pseudo labels are replaced by gray scale values 6
            labeled_images = tar[w_l, :, :]  # Labeled samples
            unlabeled_images = tar[wo_l, :, :]  # Unlabeled samples (this experiment set is 6)

            # Determine if the without_label tuple is empty
            # if no unlabeled image in this batch
            if len(wo_l) == 0:
                loss = self.criterion(probs_pred, tar)
            # if no labeled image in this batch
            elif len(w_l) == 0:
                if isinstance(probs_pred, list):
                    probs_unlabeled = torch.zeros_like(probs_pred[0][wo_l, ...])
                    for l in probs_pred:
                        probs_unlabeled += l[wo_l, ...]
                else:
                    probs_unlabeled = probs_pred[wo_l, ...]

                """
                When using strategies 1, 4, calculate the sum of kl loss and entropy minimization loss 
                When using strategies 2, 3, because the true label is 0, no reference distribution is formed and no kl loss has to be calculated 
                Whereas all four methods use an entropy minimization loss, a kl loss is added at 1, 4, and at wl = 0 is the same
                """

                if self.strategy == "Strategy1":

                    """
                    1. The prediction predicted by the pseudo label is used as the consistency regularization to compute the kl loss
                    """
                    kl_divergence_loss = self.compute_kl_loss_strategy_1(probs_unlabeled, adaptive_dist)
                    entropy_loss = self.compute_entropy_loss(probs_unlabeled)
                    total_unsupervised_loss = kl_divergence_loss + entropy_loss

                elif self.strategy == "Strategy2":
                    """
                    2. Compute the kl loss by using the prediction from the true label as a consistency regularized disturbance 
                    This does not hold here because there is no label in the first place, so there is no distribution
                    """
                    entropy_loss = self.compute_entropy_loss(probs_unlabeled)
                    total_unsupervised_loss = entropy_loss

                elif self.strategy == "Strategy3":
                    """
                    3. The true label is used as a disturbance for consistency regularization to compute the kl loss
                    No kl loss for the same reason as in 2
                    """
                    entropy_loss = self.compute_entropy_loss(probs_unlabeled)
                    total_unsupervised_loss = entropy_loss
                elif self.strategy == "Strategy4":
                    """
                    4. Comprehensive strategy
                    """
                    kl_divergence_loss = self.compute_kl_loss_strategy_1(probs_unlabeled, adaptive_dist)
                    entropy_loss = self.compute_entropy_loss(probs_unlabeled)
                    total_unsupervised_loss = kl_divergence_loss + entropy_loss
                elif self.strategy == "None":
                    """
                    5. No image-level consistency regularization
                    """
                    total_unsupervised_loss = 0

            # mixed batch with labeled and unlabeled data
            else:
                # supervised loss
                if isinstance(probs_pred, list):
                    pp = []
                    for l in probs_pred:
                        pp.append(l[w_l, ...])
                    supervised_loss = self.criterion(pp, labeled_images)
                else:
                    supervised_loss = self.criterion(probs_pred[w_l, :, :], labeled_images)

                # Obtain unlabeled and labeled predictions for subsequent use
                if isinstance(probs_pred, list):
                    probs_labeled = torch.zeros_like(probs_pred[0][w_l, ...])
                    probs_unlabeled = torch.zeros_like(probs_pred[0][w_l, ...])
                    for l in probs_pred:
                        probs_labeled += l[w_l, ...]
                        probs_unlabeled += l[wo_l, ...]
                else:
                    probs_labeled = probs_pred[w_l, ...]
                    probs_unlabeled = probs_pred[wo_l, ...]
                
                if self.strategy == "Strategy1":
                    """
                    1. Use the prediction from the pseudo label as the consistency regularization, and compute the kl loss 
                    This is the same as when wl is 0 
                    because the true label is not used as the reference distribution here.
                    """  
                    kl_divergence_loss = self.compute_kl_loss_strategy_1(probs_unlabeled, adaptive_dist)
                    entropy_loss = self.compute_entropy_loss(probs_unlabeled)
                
                elif self.strategy == "Strategy2":
                    """
                    2. Use the prediction from the ground truth label as the consistency regularization
                    """
                    entropy_loss = self.compute_entropy_loss(probs_unlabeled)
                    kl_divergence_loss = self.compute_kl_loss_strategy_2(probs_labeled, probs_unlabeled, adaptive_dist)
                    total_unsupervised_loss = kl_divergence_loss, entropy_loss

                elif self.strategy == "Strategy3":
                    """
                    3. Use the ground truth label as the consistency regularization
                    """
                    entropy_loss = self.compute_entropy_loss(probs_unlabeled)
                    kl_divergence_loss = self.compute_kl_loss_strategy_3(probs_labeled, probs_unlabeled, adaptive_dist)
                    total_unsupervised_loss = kl_divergence_loss, entropy_loss

                elif self.strategy == "Strategy4":
                    """
                    4. Compute kl loss by using the 3 diff strategies
                    """
                    entropy_loss = self.compute_entropy_loss(probs_unlabeled)

                    """
                    strategy1
                    """
                    kl_divergence_loss_1 = self.compute_kl_loss_strategy_1(probs_unlabeled, adaptive_dist)

                    """
                    strategy2
                    """
                    kl_divergence_loss_2 = self.compute_kl_loss_strategy_2(probs_labeled, probs_unlabeled, adaptive_dist)

                    """
                    strategy3
                    """
                    kl_divergence_loss_3 = self.compute_kl_loss_strategy_3(tar, w_l, probs_unlabeled)

                    total_unsupervised_loss = entropy_loss + kl_divergence_loss_1, kl_divergence_loss_2, kl_divergence_loss_3

                """
                unsupervised loss based on three different batch composition
                """
                # Calculate total loss and optimize
                loss = total_unsupervised_loss + supervised_loss

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
                   Kappa(mode="accum"))
        self.model.eval()

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

                """
                判断without_label元组是否为空
                """
                if len(w_l) == 0:
                    losses.update(0, n=1)
                else:
                    # 获取有标签和无标签数据
                    labeled_images = tar[w_l, :, :]  # 取出已知标签的样本

                    if isinstance(probs_pred, list):
                        pp = []
                        for l in probs_pred:
                            pp.append(l[w_l, ...])
                        loss = self.criterion(pp, labeled_images)
                    else:
                        loss = self.criterion(probs_pred[w_l, :, :], labeled_images)

                    losses.update(loss.item(), n=len(w_l))

                labeled_images = labeled_images

                if isinstance(prob, list):
                    prob = prob[0][w_l, ...].cpu().numpy()
                    prob_top2 = prob_top2[0][w_l, ...].cpu().numpy()
                else:
                    prob = prob[w_l, :, :].cpu().numpy()
                    prob_top2 = prob_top2[w_l, ...].cpu().numpy()


                cm = prob.astype('uint8')

                cm_top2 = prob_top2.astype('uint8')
                mask = (cm == 0)
                cm[mask] = cm_top2[mask]
                tar = tar[w_l, :, :].cpu().numpy().astype('uint8')
                mask = (tar == 0)
                cm[mask] = tar[mask]

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

                # 将4分类转换为2分类的计算
                new_cm = cm
                new_mask = (new_cm > 1)
                new_cm[new_mask] = 2
                new_tar_mask = (tar > 1)
                new_tar = tar
                new_tar[new_tar_mask] = 2

                batch_cm = confusion_matrix((new_tar[tar != 0] - 1).ravel(), (new_cm[tar != 0] - 1).ravel())
                if total_cm.shape == batch_cm.shape:
                    total_cm += batch_cm

                desc = (start_pattern + " Loss: {:.4f} ({:.4f})").format(i + 1, len_eval, losses.val, losses.avg)
                for m in metrics:
                    if isinstance(m.sum, tuple):
                        ss = ["{:.4f}".format(val) for val in m.val]
                    else:
                        ss = ["{}".format(m.val)]
                    desc += "{} {}".format(m.__name__, "___".join(ss))

                pb.set_description(desc)
                dump = not self.is_training or (i % max(1, len_eval // 10) == 0)
                if dump:
                    self.logger.dump(desc)

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

                if self.save:
                    for j in range(batch_size):
                        self.save_image(name[j], quantize(cm[j]), epoch)

            # # 为了可以产生4分类的save
            # new_cm = cm
            # new_mask = (new_cm > 1)
            # new_cm[new_mask] = 2
            # new_tar_mask = (tar > 1)
            # new_tar = tar
            # new_tar[new_tar_mask] = 2
            
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
            
            def calculate_iou(cm):
                """
                从混淆矩阵计算每个类别的 IoU
                :param cm: 混淆矩阵，形状为 (C, C)，其中 C 是类别数
                :return: 每个类别的 IoU，形状为 (C,)
                """
                num_classes = cm.shape[0]
                iou_list = []

                for c in range(num_classes):
                    # 计算 TP, FP, FN
                    TP = cm[c, c]
                    FP = np.sum(cm[:, c]) - TP  # 所有预测为 c 的样本数 - TP
                    FN = np.sum(cm[c, :]) - TP  # 所有真实为 c 的样本数 - TP

                    # 计算 IoU
                    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
                    iou_list.append(iou)

                return np.array(iou_list)

            """
            绘制confusion matrix
            """
            # def plt_metric(cm):
            #     import seaborn as sns
            #     import matplotlib.pyplot as plt
            #     labels = ["No Dmg.", "Minor Dmg.", "Major Dmg.", "Destroyed"]
            #     sns.heatmap(cm, annot=True, cmap="Blues", fmt='.3f', xticklabels=labels, yticklabels=labels,
            #                 annot_kws={"size": 20})
            #     cax = plt.gcf().axes[-1]
            #     cax.tick_params(labelsize=16)
            
            #     plt.xticks(rotation=45, fontsize=20)
            #     plt.yticks(rotation=0, fontsize=20)
            #     outpath = "/home/yujiaxi/DLModels/CDLab-SPAUNet-SSL/confusion_matrix"
            #     plt.savefig(os.path.join(outpath, "normalized_confusion_matrix_lunet.png"), bbox_inches="tight")

            
            # plt_cm_percent = plt_cm / plt_cm.sum(axis=1)[:, np.newaxis]
            # plt_metric(plt_cm_percent)

            precision = np.nan_to_num(np.diag(total_cm) / total_cm.sum(axis=0))
            print("precision:", precision)
            recall = np.nan_to_num(np.diag(total_cm) / total_cm.sum(axis=1))
            print("recall:", recall)
            f1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))
            print("f1-score:", f1)
            acc = np.nan_to_num(np.diag(total_cm).sum() / total_cm.sum())  # 计算micro的
            print("accuracy:", acc)
            kappa_index = calculate_Kappa(total_cm)
            print("Kappa-2class:", kappa_index)
            iou = calculate_iou(total_cm)
            print("IoU-2class:", iou)

            precision_4class = np.nan_to_num(np.diag(plt_cm) / plt_cm.sum(axis=0))
            print("precision-4class:", precision_4class)
            recall_4class = np.nan_to_num(np.diag(plt_cm) / plt_cm.sum(axis=1))
            print("recall-4class:", recall_4class)
            f1_4class = np.nan_to_num(2 * (precision_4class * recall_4class) / (precision_4class + recall_4class))
            print("f1-score-4class:", f1_4class)
            acc_4class = np.nan_to_num(np.diag(plt_cm).sum() / plt_cm.sum())  # 计算micro的
            print("accuracy-4class:", acc_4class)
            kappa_index_4class = calculate_Kappa(plt_cm)
            print("Kappa-4class:", kappa_index_4class)
            iou_4class = calculate_iou(plt_cm)
            print("IoU-4class:", iou_4class)

        if self.tb_on:
            self.tb_writer.add_scalar("Eval/loss", losses.avg, self.eval_step)
            for m in metrics:
                self.tb_writer.add_scalar(f"Eval/{m.__name__.lower()}", m.val, self.eval_step)
            self.tb_writer.flush()

        return metrics[3].val  # F1-score

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
class AdaptiveDistribution
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
