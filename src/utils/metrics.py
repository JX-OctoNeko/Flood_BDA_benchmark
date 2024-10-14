from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

import torch


class Meter:
    def __init__(self, callback=None, calc_avg=True):
        super().__init__()
        if callback is not None:
            self.calculate = callback
        self.calc_avg = calc_avg
        self.reset()

    def calculate(self, *args):
        if len(args) == 1:
            # print(type(args))
            # print(len(args))
            # print(args)
            return args[0]
        else:
            raise ValueError
        # if len(args) == 4:
        #     return args[0], args[1], args[2], args[3]
        # else:
        #     raise ValueError

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        if self.calc_avg:
            self.avg = 0

    def update(self, *args, n=1):
        self.val = self.calculate(*args)
        if isinstance(self.val, list):
            self.sum = sum(val for val in self.val)
        else:
            self.sum += self.val * n # 防止val是list进行的修改
        # self.sum += self.val * n  # 源代码
        self.count += n
        if self.calc_avg:
            self.avg = self.sum / self.count

    def __repr__(self):
        if self.calc_avg:
            return "val: {} avg: {} cnt: {}".format(self.val, self.avg, self.count)
        else:
            return "val: {} cnt: {}".format(self.val, self.count)


# These metrics only for numpy arrays
class Metric(Meter):
    __name__ = 'Metric'
    def __init__(self, n_classes=4, mode='separ', reduction='none'): # n_classes=4 reduction=none
        self._cm = Meter(partial(metrics.confusion_matrix, labels=np.arange(n_classes)), False)
        self.mode = mode
        if reduction == 'binary' and n_classes != 2:
            raise ValueError("Binary reduction only works in 2-class cases.")
        self.reduction = reduction
        super().__init__(None, mode!='accum')
    
    def _calculate_metric(self, cm):
        raise NotImplementedError

    def calculate(self, pred, true, n=1):
        # 为了sta和dsamnet修改
        if pred.ndim == 4:
            pred = np.sum(pred, axis=1)

        # p_has_zero = torch.any(torch.tensor(pred == 0))
        # p_has_one = torch.any(torch.tensor(pred == 1))
        # p_has_two = torch.any(torch.tensor(pred == 2))
        # p_has_three = torch.any(torch.tensor(pred == 3))
        # p_has_four = torch.any(torch.tensor(pred == 4))
        #
        # # 打印结果
        # print("pred含有灰度值 0 的像素：", p_has_zero)
        # print("pred含有灰度值 1 的像素：", p_has_one)
        # print("pred含有灰度值 2 的像素：", p_has_two)
        # print("pred含有灰度值 3 的像素：", p_has_three)
        # print("pred含有灰度值 4 的像素：", p_has_four)

        # has_zero = torch.any(torch.tensor(true == 0))
        # has_one = torch.any(torch.tensor(true == 1))
        # has_two = torch.any(torch.tensor(true == 2))
        # has_three = torch.any(torch.tensor(true == 3))
        # has_four = torch.any(torch.tensor(true == 4))
        #
        # # 打印结果
        # print("true含有灰度值 0 的像素：", has_zero)
        # print("true含有灰度值 1 的像素：", has_one)
        # print("true含有灰度值 2 的像素：", has_two)
        # print("true含有灰度值 3 的像素：", has_three)
        # print("true含有灰度值 4 的像素：", has_four)

        true = true.ravel()
        pred = pred.ravel()
        # print(true[true!=0])
        # print(pred[true!=0])

        # self._cm.update(true.ravel(), pred.ravel())
        self._cm.update(true[true!=0]-1, pred[true!=0]-1)

        # print(self._cm.sum)
        # print(self._cm.val)

        if self.mode == 'accum':
            cm = self._cm.sum
        elif self.mode == 'separ':
            cm = self._cm.val
        else:
            raise ValueError("Invalid working mode")

        if self.reduction == 'none':
            # Do not reduce size
            return self._calculate_metric(cm)
        elif self.reduction == 'mean':
            # Macro averaging
            return self._calculate_metric(cm).mean()
        elif self.reduction == 'binary':
            # The pos_class be 1
            return self._calculate_metric(cm)[1]
        else:
            raise ValueError("Invalid reduction type")

    def reset(self):
        super().reset()
        # Reset the confusion matrix
        self._cm.reset()

    def __repr__(self):
        return self.__name__+" "+super().__repr__()


class Precision(Metric):
    __name__ = 'Prec.'
    def _calculate_metric(self, cm):
        return np.nan_to_num(np.diag(cm)/cm.sum(axis=0))



class Recall(Metric):
    __name__ = 'Recall'
    def _calculate_metric(self, cm):
        return np.nan_to_num(np.diag(cm)/cm.sum(axis=1))



class Accuracy(Metric):
    __name__ = 'OA'
    def __init__(self, n_classes=4, mode='separ'): # 改成4类
        super().__init__(n_classes=n_classes, mode=mode, reduction='none')
        
    def _calculate_metric(self, cm):
        return np.nan_to_num(np.diag(cm).sum()/cm.sum())

    # def _calculate_metric(self, confusion_matrix):
    #     num_classes = len(confusion_matrix)
    #     accuracies = []
    #
    #     for i in range(num_classes):
    #         tp = confusion_matrix[i][i]  # True positive
    #         fp = np.sum(confusion_matrix[:, i]) - tp  # False positive
    #         fn = np.sum(confusion_matrix[i, :]) - tp  # False negative
    #         tn = np.sum(confusion_matrix) - tp - fp - fn  # True negative
    #
    #         if (tp + fp + fn + tn) == 0:
    #             accuracy = 0
    #         else:
    #             accuracy = tp / (tp + fp)
    #
    #         accuracies.append(accuracy)
    #
    #     macro_accuracy = np.mean(accuracies)
    #     return macro_accuracy # 计算总体macro_accuracy

    # def _calculate_metric(self, confusion_matrix):
    #     num_classes = len(confusion_matrix)
    #     accuracies = []
    #
    #     for i in range(num_classes):
    #         tp = confusion_matrix[i][i]  # True positive
    #         fp = np.sum(confusion_matrix[:, i]) - tp  # False positive
    #         fn = np.sum(confusion_matrix[i, :]) - tp  # False negative
    #         tn = np.sum(confusion_matrix) - tp - fp - fn  # True negative
    #
    #         sample_count = np.sum(confusion_matrix)
    #         print(sample_count)
    #
    #         if (tp + fp + fn + tn) == 0:
    #             accuracy = 0
    #         else:
    #             accuracy_1 = (tp + tn) / (tp + tn + fn + fp)
    #             accuracy = ((tp + fn) / sample_count) * accuracy_1
    #
    #         accuracies.append(accuracy)
    #
    #     return np.sum(accuracies) # 计算分别的acc




class F1Score(Metric):
    __name__ = 'F1'
    def _calculate_metric(self, cm):
        prec = np.nan_to_num(np.diag(cm)/cm.sum(axis=0))
        recall = np.nan_to_num(np.diag(cm)/cm.sum(axis=1))
        return np.nan_to_num(2*(prec*recall) / (prec+recall))

class kappa(Metric):
    __name__ = "Kappa"

    def __init__(self, n_classes=4, mode='separ'):
        super().__init__(n_classes=n_classes, mode=mode, reduction='none')

    # def plt_metric(self, cm):
    #
    #     import seaborn as sns
    #     import matplotlib.pyplot as plt
    #     labels = ["No damage", "Minor damage", "Major damage", "Destroyed"]
    #     fig, ax = plt.subplot()
    #     sns.heatmap(cm, annot=True, cmap="Blue", fmt='d', xticklabels=labels, yticklabels=labels)
    #     ax.set_title("Confusion Matrix")
    #     ax.set_xlabel("Predicted Labels")
    #     ax.set_ylabel("True Labels")
    #
    #     plt.show()

    def _calculate_metric(self, cm):
        p_o = np.nan_to_num(np.diag(cm).sum() / cm.sum())

        pe_rows = cm.sum(axis=0)
        pe_cols = cm.sum(axis=1)
        sum_total = sum(pe_cols)
        p_e = np.nan_to_num(np.dot(pe_rows, pe_cols) / float(sum_total ** 2))

        kappa = (p_o - p_e) / (1 - p_e)

        return kappa

