import cv2
import numpy as np
import torch
import torch.nn as nn

from ActivationsAndGradients import ActivationsAndGradients


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model = self.model.cuda()
        else:
            pass
        self.activations_and_grads = ActivationsAndGradients(self.model,
                                                           target_layers, reshape_transform)
        # 实例化ActivationAndGradients类

    @staticmethod
    def get_loss(output, target):
        loss = output # 直接将预测作为loss回传，这是语义分割的结果
        return loss

    @staticmethod
    def get_cam_weights(grads):
        # GAP global average pooling, get[B,C,1,1]
        # because only one input graph, so B=1, C is channels
        return np.mean(grads, axis=(2,3), keepdims=True)

    @staticmethod
    def get_target_width_height(input_tensor):
        # get original graph width and height
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def get_cam_image(self, activations, grads):
        # 将梯度图进行GAP， weights大小是[1,C,1,1]在通道上具有不同权重分布
        weights = self.get_cam_weights(grads) # 对梯度图进行全局平均池化
        weighted_activations = weights * activations # 和原特征层进行加权乘法
        cam = weighted_activations.sum(axis=1) # 在C维度上求和，得到大小为(1,h,w)
        return cam

    @staticmethod
    def scale_cam_img(cam, target_size=None):
        # 将cam缩放到原始图像相同大小，并将其值缩放到[0,1]之间
        result = []
        for img in cam:
            # 因为传入的目标层 target_layers可能是复数，所以一层层看
            img = img - np.min(img) # 减去最小值
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
                # 注意cv2.resize(src, (width, height))是width在前
            result.append(img)
        result = np.float32(result)
        return result

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy() for a in
                            self.activations_and_grads.activations]
        grads_list = [a.cpu().data.numpy() for a in
                      self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)
        cam_per_target_layer = []

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            # 一张一张特征图和梯度对应着处理
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam<0] = 0 #ReLU
            scaled = self.scale_cam_img(cam, target_size)
            # 将CAM图缩放到原图大小，然后与原图叠加，这考虑到特征图可能小于或大于原图情况
            cam_per_target_layer.append(scaled[:, None, :])
             # 在None标注的位置加入一个维度，相当于scaled.unsqueeze(1)，此时scaled大小为
             # [1,1,H,W]
        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_layer):
        cam_per_layer = np.concatenate(cam_per_layer, axis=1)
        # 在channel维度进行堆叠，并不相加处理
        cam_per_layer = np.maximum(cam_per_layer, 0)
        # 当cam_per_layer任意位置小于0，就变为0
        result = np.mean(cam_per_layer, axis=1)
        # 在channels维度求平均，压缩这个维度，让该维度返回1
        # 如果输入的是多层网络结构，会将这些结构在channel维度上压缩，形成一张图
        return self.scale_cam_img(result)

    def __call__(self, input_tensor_1, input_tensor_2, target, *args, **kwargs): # __init__()后自动调用call方法
        # 这里target就是目标的gt
        if self.use_cuda:
            input_tensor_1 = input_tensor_1.cuda()
            # 正向传播输出结果，创建ActivationsAndGradients类后调call方法，执行self.model(x)
            # 这里output未经过softmax，如果网络结构中最后的output不能经历激活函数
            output = self.activations_and_grads(input_tensor_1, input_tensor_2)[0]
            _output = output.detach().cpu()
            _output = _output.squeeze(0).squeeze(0)

            self.model.zero_grad()
            loss = self.get_loss(output, target)
            loss.backward(torch.ones_like(target), retain_graph=True)
            # 将输出结果作为loss回传，并且记得梯度图
            # 梯度最大说明该层特征在预测过程中起到作用最大
            # 预测部分展示出来的就是注意力

            cam_per_layer = self.compute_cam_per_layer(input_tensor_1)
            # 计算每一层指定网络结构中的cam
            return self.aggregate_multi_layers(cam_per_layer)
            # 将指定的层结构中得到的cam图堆叠一张图

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_val, IndexError):
            # handel index error here
            print(
                f"An exception occured in CAM with block: {exc_type}. Message: {exc_val}"
            )
            return True

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap) #将cam的结果转成伪彩色图片
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) #使用opencv方法后，得到的一般都是BGR格式，还要转化为RGB格式
        # OpenCV中图像读入的数据格式是numpy的ndarray数据格式。是BGR格式，取值范围是[0,255].
    heatmap = np.float32(heatmap) / 255. #缩放到[0,1]之间

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255*cam)