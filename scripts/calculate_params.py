import torch
from src.models import unet, cdnet, siamunet_conc, siamunet_diff, lunet, bit, p2v, snunet
from thop import profile

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# 加载我的模型
bit_model = bit.BIT(in_ch=3,out_ch=5,backbone='resnet18',n_stages=4,token_len=4,enc_with_pos=True,enc_depth=1,dec_depth=8,dec_head_dim=8)

snu_model = snunet.SNUNet(3,5,32)
p2v_model = p2v.P2VNet(
    in_ch=3,
    video_len=8
)
unet_model = unet.UNet(in_ch=6, out_ch=5)
fcdiff_model = siamunet_diff.SiamUNet_diff(in_ch=3, out_ch=5)
fcconc_model = siamunet_conc.SiamUNet_conc(in_ch=3, out_ch=5)
lunet_model = lunet(in_ch=3, out_ch=5)
cdnet_model = cdnet(in_ch=6, out_ch=5)


ckp_path = "checkpoint_latest_siamunet-conc.pth"

def print_model_params(model, ckp_path):
    # checkpoint = torch.load('checkpoint_latest_snunet.pth')
    # checkpoint = torch.load('checkpoint_latest_bit.pth')
    checkpoint = torch.load(ckp_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    input_tensor = torch.randn(2,3,256,256)
    flops, params = profile(model, inputs=(input_tensor,))

    # print(f"The model has {count_parameters(model):,} parameters.")
    # 输出结果
    print(f"FLOPs: {flops}")
    print(f"GFLOPs: {flops / 1e9:.2f}")
    print(f"参数量: {params}")

print_model_params(fcconc_model, ckp_path)