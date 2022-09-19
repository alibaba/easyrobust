import requests
import PIL
import io
from torchvision import transforms
from easyrobust.third_party.vqgan import VQModel, reconstruct_with_vqgan

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])

# # vqgan used by drvit
# ddconfig = {'double_z': False, 'z_channels': 256, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1,1,2,2,4], 'num_res_blocks': 2, 'attn_resolutions':[16], 'dropout': 0.0}
# vqgan = VQModel(ddconfig, n_embed=1024, embed_dim=256, ckpt_path='http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/pretrained_models/vqgan_imagenet_f16_1024.ckpt')

# vqgan used by dat
ddconfig = {'double_z': False, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1,2,2,4], 'num_res_blocks': 2, 'attn_resolutions':[32], 'dropout': 0.0}
vqgan = VQModel(ddconfig, n_embed=16384, embed_dim=4, ckpt_path='http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/pretrained_models/vqgan_openimages_f8_16384.ckpt')

img = download_image('http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/assets/test.png')
xrec = reconstruct_with_vqgan(transform(img).unsqueeze(0).cuda(), vqgan.cuda())
PIL.Image.fromarray((xrec[0].permute(1,2,0).detach().cpu().numpy()*255.).astype('uint8')).save('tmp.png')


