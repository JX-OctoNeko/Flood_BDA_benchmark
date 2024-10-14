import torch

from .cd_trainer import CDTrainer


class CDTrainer_BCE(CDTrainer):
    def _prepare_data(self, t1, t2, tar):
        return super()._prepare_data(t1, t2, tar.float())

    def _process_model_out(self, out):
        return out.squeeze(1)

    def _pred_to_prob(self, out):
        print("输入的pred是", type(out))
        return [torch.nn.functional.softmax(o, dim=1).squeeze(1) for o in out]
        # return torch.nn.functional.softmax(out, dim=1) # 这里的源代码是sigmoid 我修改成softmax