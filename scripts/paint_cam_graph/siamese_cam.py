import cv2
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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


# 设置eval，固定batchnormal和dropout等模型过程中的处理
print(model)
ckpt_path = osp.join(osp.dirname(osp.abspath(__file__)), 'checkpoint', 'checkpoint_latest_bit_60.pth')
state_dict = torch.load(ckpt_path)['state_dict']
model.load_state_dict(state_dict)
model.eval().to(device)

# 准备储存过程中的特征图和梯度图
grad_block = []
feature_block = []

# module选择模块，grad_in:module 的输入；grad_out:module 的输出
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach()) # grad_out is a tuple, detach can only use tensor

def forward_hook(module, input, output):
    feature_block.append(output)

# ####### 这里是选择层的地方
# # 选择自己网络中要勾选的模块
# model.classifier[0].seq[-1].register_forward_hook(forward_hook)
# # 正向传播的时候自动执行，和反向传播的时候子自动执行
# model.classifier[0].seq[-1].register_full_backward_hook(backward_hook)


# for name, module in model.named_modules():
#     if isinstance(module, nn.Conv2d):
#         # print(name)
#         key_name = str(module.weight.shape)
#
#         module_name_list = name.split(".")
#         print(module_name_list)
#         if len(module_name_list) == 3:
#             n1 = module_name_list[0]
#             n2 = module_name_list[1]
#             n3 = module_name_list[2]
#             model._modules[n1]._modules[n2]._modules[n3].register_forward_hook(forward_hook)
#             model._modules[n1]._modules[n2]._modules[n3].register_full_backward_hook(backward_hook)
#         elif len(module_name_list) == 4:
#             n1 = module_name_list[0]
#             n2 = module_name_list[1]
#             n3 = module_name_list[2]
#             n4 = module_name_list[3]
#             model._modules[n1]._modules[n2]._modules[n3]._modules[n4].register_forward_hook(forward_hook)
#             model._modules[n1]._modules[n2]._modules[n3]._modules[n4].register_full_backward_hook(backward_hook)
#         elif len(module_name_list) == 5:
#             n1 = module_name_list[0]
#             n2 = module_name_list[1]
#             n3 = module_name_list[2]
#             n4 = module_name_list[3]
#             n5 = module_name_list[4]
#             model._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules[n5].register_forward_hook(forward_hook)
#             model._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules[n5].register_full_backward_hook(
#                 backward_hook)
#         elif len(module_name_list) == 6:
#             n1 = module_name_list[0]
#             n2 = module_name_list[1]
#             n3 = module_name_list[2]
#             n4 = module_name_list[3]
#             n5 = module_name_list[4]
#             n6 = module_name_list[5]
#             model._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules[n5]._modules[n6].register_forward_hook(forward_hook)
#             model._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules[n5]._modules[n6].register_full_backward_hook(
#                 backward_hook)

def register_hooks(model, module_name_list):
    target_module = model
    for name in module_name_list:
        target_module = target_module._modules[name]

    target_module.register_forward_hook(forward_hook)
    target_module.register_full_backward_hook(backward_hook)

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d): # 注意这里可以更改对象层
        module_name_list = name.split(".")
        # print(module_name_list)
        register_hooks(model, module_name_list)

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
    folder_path = osp.join(osp.dirname(osp.abspath(__file__)), 'test_img')
    pre_folder_path = osp.join(folder_path, 'pre')
    post_folder_path = osp.join(folder_path, 'post')
    target_folder_path = osp.join(folder_path, 'label')

    pre_file_path = osp.join(pre_folder_path, file_name)
    post_file_path = osp.join(post_folder_path, file_name)
    target_file_path = osp.join(target_folder_path, file_name)

    return pre_file_path, post_file_path, target_file_path



pre_img_path, post_img_path, target_img_path = img_path('midwest-flooding_00000010_disaster_2.png')

pre_img_pil = Image.open(pre_img_path)
post_img_pil = Image.open(post_img_path)

pre_input_tensor = preprocessing_img(pre_img_pil).unsqueeze(0).to(device)
post_input_tensor = preprocessing_img(post_img_pil).unsqueeze(0).to(device)

target_img_pil = Image.open(target_img_path)
target_transform = transforms.Compose([transforms.Resize(224),
                                       transforms.ToTensor()])
target_input_tensor = target_transform(target_img_pil).unsqueeze(0).to(device)

def get_cam(input_tensor1, input_tensor2, i, save_path):

    # 正向传播
    output = model(input_tensor1, input_tensor2)

    # 网络中没有激活层就需要这一行
    output = torch.softmax(output, dim=2)

    # 得到概率最大输出下标
    # cls_scores = output.view(output.shape[1]*output.shape[0], -1)
    # print(cls_scores.shape)
    # _, max_idx = torch.max(cls_scores.detach().cpu(), 1)
    # # max_idx = 1
    # print(max_idx)

    # 清空模型参数
    model.zero_grad()

    # 获得概率最大下标对应的概率（不是概率值，概率值再torch.max中可以得到，一定是output，不然无法求导）
    loss = output.mul(target_input_tensor)
    _loss = loss.detach().cpu()
    _loss = _loss.squeeze(0).squeeze(0)
    # print(loss.shape)

    # 得到影响概率值的梯度
    loss_mean = loss.mean()
    loss_mean.backward()

    # 保证形状传参一样
    pre_img = pre_input_tensor[0]
    pre_img = pre_img.cpu().detach().numpy()
    pre_img = pre_img.transpose((1,2,0))

    # pre_img一样
    post_img = post_input_tensor[0]
    post_img = post_img.cpu().detach().numpy()
    post_img = post_img.transpose((1, 2, 0))

    print(grad_block[i].shape)
    # 保证grad_和features_是一样的[C,H,W]
    grads_pre = grad_block[i].cpu().data.numpy().squeeze()
    fmap_pre = feature_block[i].cpu().data.numpy().squeeze()
    # 传入原始图片，特征图，梯度图，开始计算cam
    cam_img = cam_show_img(pre_img, fmap_pre, grads_pre, save_path)

    return cam_img


def cam_show_img(img, feature_maps, grads, save_path):
    # img [H,W,3]
    # fmap [C,H,W]
    # grads [C,H,W]

    # cam [H,W]
    cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)

    # grad [C,H*W]
    grads = grads.reshape([feature_maps.shape[0], -1]) # reshape成feature maps

    # 根据梯度图平均池化得到梯度向量[1, H*W]
    weights = np.mean(grads, axis=1)

    # 再特征图对应维度加权然后所有维度求和
    for i, w in enumerate(weights):
        cam += w * feature_maps[i, :, :]

    # 去除负值 relu 并且归一化，方便还原成0-255
    # cam [H,W]
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    # 将cam图resize原图大小
    cam = cv2.resize(cam, (img.shape[0], img.shape[1]))

    # cam制作成三通道上色 cam [H,W,3]
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # BGR - RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 把tensor还原成图片
    cam_img = 0.3 * heatmap + 0.7 * img

    # 第一个plt展示cam+img
    ax1 = plt.subplot(1,2,1)
    plt.axis('off')
    ax1.imshow(cam_img/255)

    ## 第二个plt展示cam
    ax2 = plt.subplot(1, 2, 2)
    plt.axis('off')
    ax2.imshow(heatmap / 255)

    # # 保存图片到指定文件夹
    # plt.savefig(save_path)
    plt.show()

    return cam_img / 255


for i in range(36): # 这个取决于grad_block中储存了多少
    save_path = r'C:\Users\student\Desktop\半监督结果\visual_result\cam\cam_img_{}.png'.format(i)
    cam_img = get_cam(pre_input_tensor, post_input_tensor, i, save_path)



