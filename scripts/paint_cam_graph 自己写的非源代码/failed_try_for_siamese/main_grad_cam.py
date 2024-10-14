import os.path as osp
import numpy as np

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from cam_utils import GradCAM, show_cam_on_image
from src.models.bit import BIT


# 有 GPU 就用 GPU，没有就用 CPU
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
model.eval()


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
target_img_pil = Image.open(target_img_path)

pre_input_tensor = preprocessing_img(pre_img_pil).unsqueeze(0)
post_input_tensor = preprocessing_img(post_img_pil).unsqueeze(0)

target_transform = transforms.Compose([transforms.Resize(224),
                                       transforms.ToTensor()])
target_input_tensor = target_transform(target_img_pil).unsqueeze(0)

target_layers = [model.backbone.conv_out.seq]

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
grayscale_cam = cam(input_tensor_1=pre_input_tensor, input_tensor_2=post_input_tensor, target=target_input_tensor)
print(grayscale_cam)


# grayscale_cam = grayscale_cam[0, :]
# visualization = show_cam_on_image(pre_img_path.astype(dtype=np.float32) / 255.,
#                                       grayscale_cam,
#                                       use_rgb=True)
# plt.imshow(visualization)
# plt.show()