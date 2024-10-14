import torch

from .cd_trainer import CDTrainer


class CDTrainer_NLL(CDTrainer):
    def _process_model_out(self, out):
        return out.squeeze(1)

    def _pred_to_prob(self, pred):
        # return torch.exp(pred, dim=1)
        # 梯度裁剪
        # 检查张量中是否有NaN
        contains_nan = torch.isnan(pred).any()

        # 或者同时检查NaN和Inf
        contains_invalid = torch.isinf(pred) | torch.isnan(pred)
        contains_invalid_any = contains_invalid.any()

        # if contains_nan:
        #     print("Tensor中含有NaN值")
        #
        # if contains_invalid_any:
            # print("Tensor中含有NaN或Inf值")

        # 如果只需要知道哪些元素是NaN或Inf，可以进一步获取其索引
        nan_indices = torch.nonzero(torch.isnan(pred))
        inf_indices = torch.nonzero(torch.isinf(pred))
        # print(f"是否含有nan值：{nan_indices}")
        # print(f"是否含有inf值：{inf_indices}")

        # if torch.isnan(torch.nn.functional.softmax(pred, dim=1)).any():
             # print(f"结果含有nan值！")

        # return torch.nn.functional.log_softmax(pred, dim=1)

        return torch.nn.functional.softmax(pred, dim=1)



