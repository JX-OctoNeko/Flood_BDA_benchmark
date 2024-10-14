import os

import numpy as np
import cv2
from PIL import Image
import os.path as osp

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.utils as vutils
from torchvision import transforms
from torch import optim
from torchvision.models import alexnet
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("../logs")

from src.models.bit import BIT

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('device', device)

model = BIT(
    in_ch=3,
    out_ch=2,
    backbone='resnet18',
    n_stages=4,
    token_len=4,
    enc_with_pos=True,
    enc_depth=1,
    dec_depth=8,
    dec_head_dim=8
)


# print(model)
ckpt_path = osp.join(osp.dirname(osp.abspath(__file__)), '../checkpoint', 'checkpoint_latest_bit.pth')
state_dict = torch.load(ckpt_path)['state_dict']
model.load_state_dict(state_dict)
model.eval().to(device)


def preprocessing_img(img_pil):
    # 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
    test_transform = transforms.Compose([transforms.Resize(224),
                                         # transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.2816, 0.3242, 0.2145],
                                             std=[0.1112, 0.0934, 0.0868])
                                        ])

    return test_transform(img_pil)


def img_path(file_name):
    folder_path = osp.join(osp.dirname(osp.abspath(__file__)), '../test_img')
    pre_folder_path = osp.join(folder_path, 'pre')
    post_folder_path = osp.join(folder_path, 'post')
    target_folder_path = osp.join(folder_path, 'label')

    pre_file_path = osp.join(pre_folder_path, file_name)
    post_file_path = osp.join(post_folder_path, file_name)
    target_file_path = osp.join(target_folder_path, file_name)

    return pre_file_path, post_file_path, target_file_path



pre_img_path, post_img_path, target_img_path = img_path('midwest-flooding_00000373_disaster_0.png')

pre_img_pil = Image.open(pre_img_path)
post_img_pil = Image.open(post_img_path)

pre_input_tensor = preprocessing_img(pre_img_pil).unsqueeze(0).to(device)
post_input_tensor = preprocessing_img(post_img_pil).unsqueeze(0).to(device)


def show_cam_on_image(img, mask, out_dir):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    path_cam_img = os.path.join(out_dir, "cam.jpg")
    path_raw_img = os.path.join(out_dir, "raw.jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    cv2.imwrite(path_raw_img, np.uint8(255 * img))


def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, index+1).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605

    return class_vec


def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

    weights = np.mean(grads, axis=(1, 2))  #

    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (32, 32))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam



# 创建两个空字典用来储存feature map和grad
fmap_dict = dict()
grad_dict = dict()

# 注册hook功能
def forward_hook_func(module, input, output):
    key_name = str(module.weight.shape)
    fmap_dict[key_name].append(output)  # 索引名字，添加特征图
    # print(f"fmap_dict:{fmap_dict}")

# 后向传播
def backward_hook_func(module, grad_in, grad_out):
    key_name = str(module.weight.shape)
    grad_dict[key_name].append(grad_out[0].detach())


target_img_pil = Image.open(target_img_path)
target_transform = transforms.Compose([transforms.Resize(224),
                                       transforms.ToTensor()])
target_input_tensor = target_transform(target_img_pil).unsqueeze(0).to(device)


for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        # print(name)
        key_name = str(module.weight.shape)

        # print(key_name)
        fmap_dict.setdefault(key_name, list())
        # print(f"fmap_dict:{fmap_dict}") # 构建key value对
        grad_dict.setdefault(key_name, list()) # 初始化grad

        module_name_list = name.split(".")
        print(module_name_list)
        if len(module_name_list) == 3:
            n1 = module_name_list[0]
            n2 = module_name_list[1]
            n3 = module_name_list[2]
            model._modules[n1]._modules[n2]._modules[n3].register_forward_hook(forward_hook_func)
            model._modules[n1]._modules[n2]._modules[n3].register_full_backward_hook(backward_hook_func)
        elif len(module_name_list) == 4:
            n1 = module_name_list[0]
            n2 = module_name_list[1]
            n3 = module_name_list[2]
            n4 = module_name_list[3]
            model._modules[n1]._modules[n2]._modules[n3]._modules[n4].register_forward_hook(forward_hook_func)
            model._modules[n1]._modules[n2]._modules[n3]._modules[n4].register_full_backward_hook(backward_hook_func)
        elif len(module_name_list) == 5:
            n1 = module_name_list[0]
            n2 = module_name_list[1]
            n3 = module_name_list[2]
            n4 = module_name_list[3]
            n5 = module_name_list[4]
            model._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules[n5].register_forward_hook(forward_hook_func)
            model._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules[n5].register_full_backward_hook(
                backward_hook_func)
        elif len(module_name_list) == 6:
            n1 = module_name_list[0]
            n2 = module_name_list[1]
            n3 = module_name_list[2]
            n4 = module_name_list[3]
            n5 = module_name_list[4]
            n6 = module_name_list[5]
            model._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules[n5]._modules[n6].register_forward_hook(forward_hook_func)
            model._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules[n5]._modules[n6].register_full_backward_hook(
                backward_hook_func)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# forward
output = model(pre_input_tensor, post_input_tensor)
# print(output.shape)
# print(fmap_dict)
# idx = np.argmax(output.cpu().data.numpy())
# print("predict: {}".format(classes[idx]))

# backward
model.zero_grad()
# loss = output
# loss.backward(torch.ones_like(target_input_tensor), retain_graph=True)
# class_loss = comp_class_vec(output)
# class_loss.backward()



# add image
# for layer_name, fmap_list in fmap_dict.items(): # 返回一个可迭代列表
#     for grad_name, grad_list in grad_dict.items():
#         fmap = fmap_list[0] # 将list元素取出来,feature maps
#         fmap.transpose_(0,1) # 转置0和1维度
#         grad = grad_list[0]
#
#         # 生成cam
#         grads_val = grad.cpu().data.numpy().squeeze()
#         fmap = fmap.cpu().data.numpy().squeeze()
#         cam = gen_cam(fmap, grads_val)
#
#         output_dir = r"C:\Users\student\DLModels\cdlab\CDLab\paint_cam_graph\test_img"
#         # 保存cam图片
#         img_show = np.float32(cv2.resize(pre_img_path, (32, 32))) / 255
#         show_cam_on_image(img_show, cam, output_dir)

        # tensorboard显示
        # nrow = int(np.sqrt(fmap.shape[0]))
        # fmap_grid = vutils.make_grid(fmap, normalize=True, scale_each=True, nrow=nrow)
        # writer.add_image('feature map in {}'.format(layer_name), fmap_grid, global_step=0)


