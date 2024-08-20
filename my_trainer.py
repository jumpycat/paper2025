import argparse
import os
import sys
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.nn.utils import clip_grad_norm_
import time
import utils
import utils_img
import utils_model
from torch import optim
from torch import Tensor
import random
from datetime import timedelta

# sys.path.append('src')
import data_augmentation

from torch.nn.modules.batchnorm import _BatchNorm

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

def read_txt_to_dict(file_path):
    result_dict = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
                result_dict[key] = value
    return result_dict


def format_time(seconds):
    delta = timedelta(seconds=seconds)
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{days}d {hours}h {minutes}m {seconds}s'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Data parameters')
    aa("--train_dir", type=str, default='coco_train/train_2017')

    group = parser.add_argument_group('Model parameters')
    aa("--ldm_config", type=str, default="v2-inference.yaml", help="Path to the configuration file for the LDM model")
    aa("--ldm_stu_config", type=str, default="v2-inference_stu.yaml", help="Path to the configuration file for the LDM model")
    aa("--ldm_ckpt", type=str, default="768-v-ema.ckpt", help="Path to the checkpoint file for the LDM model")

    group = parser.add_argument_group('Training parameters')
    aa("--batch_size", type=int, default=6, help="Batch size for training")
    aa("--img_size", type=int, default=512, help="Resize images to this size")
    aa("--loss_i", type=str, default="watson-vgg", help="Type of loss for the image loss. Can be watson-vgg, mse, watson-dft, etc.")
    aa("--loss_w", type=str, default="bce", help="Type of loss for the watermark loss. Can be mse or bce")
    aa("--lambda_i", type=float, default=10.0, help="Weight of the image loss in the total loss")
    aa("--lambda_w", type=float, default=0.01, help="Weight of the watermark loss in the total loss")
    aa("--lr", type=float, default=3e-4)
    aa("--steps", type=int, default=100000, help="Number of steps to train the model for")


    group = parser.add_argument_group('Logging and saving freq. parameters')
    aa("--log_freq", type=int, default=10, help="Logging frequency (in steps)")
    aa("--save_img_freq", type=int, default=1000, help="Frequency of saving generated images (in steps)")
    aa("--save_model_freq", type=int, default=10000, help="Frequency of saving generated images (in steps)")

    group = parser.add_argument_group('Experiments parameters')
    aa("--seed", type=int, default=42)
    aa("--bits", type=int, default=48)
    aa("--warm_steps", type=int, default=10000)
    aa("--temp", type=float, default=1.0)
    aa("--cum_times", type=float, default=10)
    aa("--resume", type=str, default='240810104030/models/checkpoint_010000.pth')
    aa("--if_tanh",  type=float, default=0.0)

    group = parser.add_argument_group('DA parameters')
    aa("--if_atk", type=float, default=1.0)
    aa("--attack", type=str,default='gbcnhja')

    return parser 

def main(params):
    # Create directories if not exist.
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%y%m%d%H%M%S")

    exp_path = os.path.join('runs',dt_string)
    model_save_dir = os.path.join('runs',dt_string,'models')
    sample_dir = os.path.join('runs',dt_string,'samples')
    os.makedirs(exp_path,exist_ok=True)
    os.makedirs(model_save_dir,exist_ok=True)
    os.makedirs(sample_dir,exist_ok=True)

    cmd = ' '.join(sys.argv)
    with open(os.path.join(exp_path,'args.txt'), 'a') as file:
        for arg in vars(params):
            file.write(f"{arg}: {getattr(params, arg)}\n")
        file.write(f"{cmd}\n")


    # Set seeds for reproductibility 
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)
    
    
    # Loads the data
    vqgan_transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.RandomCrop(params.img_size),
        transforms.ToTensor(),
        utils_img.normalize_vqgan])

    Minus112ZeroOne = transforms.Normalize(mean=[-1, -1, -1], std=[1/0.5, 1/0.5, 1/0.5])
    from torchvision.datasets import ImageFolder
    train_dataset = ImageFolder(params.train_dir, transform=vqgan_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=8, drop_last=True)

    # Loads LDM auto-encoder models pretrained
    config = OmegaConf.load(f"{params.ldm_config}")
    ldm_ae  = utils_model.load_model_from_config(config, params.ldm_ckpt).first_stage_model  # LatentDiffusion -> AutoencoderKL
    ldm_ae.eval()
    ldm_ae.to(device)
    
    # Loads student
    config_stu = OmegaConf.load(f"{params.ldm_stu_config}")
    stu = utils_model.load_model_from_config(config_stu, params.ldm_ckpt).first_stage_model
    stu.encoder = nn.Identity()
    stu.quant_conv = nn.Identity()
    stu.to(device)
    stu.decoder.train()

    # Loads hidden decoder
    from efficientnet_pytorch import EfficientNet
    wm_decoder = EfficientNet.from_pretrained('efficientnet-b0')
    # wm_decoder = EfficientNet.from_name('efficientnet-b0')
    feature = wm_decoder._fc.in_features
    wm_decoder._fc = nn.Linear(in_features=feature, out_features=params.bits, bias=True)
    wm_decoder.to(device)
    wm_decoder.train()

    from sam import SAM
    base_optimizer = optim.Adam
    optimizer = SAM(list(stu.parameters()) + list(wm_decoder.parameters()), base_optimizer, lr=params.lr)

    try:
        weights = torch.load(params.resume)
        wm_decoder.load_state_dict(weights['wm_decoder'], strict=True)
        stu.load_state_dict(weights['ldm_decoder'],strict=True)
        print('Weights Loaded!')
        start_step = 20001

        optimizer_state_dict = torch.load(weights['optimizer'], strict=True)
        optimizer.base_optimizer.load_state_dict(optimizer_state_dict)

    except:
        pass

    for param in [*ldm_ae.parameters()]:
        param.requires_grad = False
    for param in [*stu.parameters(), *wm_decoder.parameters()]:
        param.requires_grad = True


    # Create losses
    import lpips
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    my_mse = lambda imgs_w, imgs: torch.mean((imgs_w - imgs)**2)
    loss_w_1 = lambda decoded, keys: F.binary_cross_entropy_with_logits(decoded * 1, keys, reduction='mean')
    loss_w_10 = lambda decoded, keys: F.binary_cross_entropy_with_logits(decoded * 10, keys, reduction='mean')


    img_ret = []
    wm_ret = []
    train_iter = iter(train_loader)
    start_time = time.time()
    total_steps = 1

    accumulated_grads_stu_history = {name: torch.zeros_like(param) for name, param in stu.named_parameters()}
    accumulated_grads_de_history = {name: torch.zeros_like(param) for name, param in wm_decoder.named_parameters()}
    
    for _ in range(params.steps + 1):

        watermark = torch.zeros((params.batch_size, params.bits), dtype=torch.float).random_(0, 2).to(device)

        accumulated_grads_stu = {name: torch.zeros_like(param) for name, param in stu.named_parameters()}
        accumulated_grads_de = {name: torch.zeros_like(param) for name, param in wm_decoder.named_parameters()}

        cloned_state_dict_stu = {name: param.clone() for name, param in stu.named_parameters()}
        cloned_state_dict_de = {name: param.clone() for name, param in wm_decoder.named_parameters()}

        for _ in range(params.cum_times):
            total_steps += 1
            try:
                imgs = next(train_iter)[0].to(device)
            except:
                train_iter = iter(train_loader)
                imgs = next(train_iter)[0].to(device)


            # encode images
            with torch.no_grad():
                imgs_z = ldm_ae.encode(imgs) 
                imgs_z = imgs_z.mode()


            enable_running_stats(stu)
            enable_running_stats(wm_decoder)

            imgs_marked = stu.decode(imgs_z, watermark) # b z h/f w/f -> b c h w
            lossi_old = params.lambda_i * loss_fn_vgg.forward(imgs, imgs_marked).mean()
            if params.if_tanh:
                decoded_old = wm_decoder((imgs_marked))
            else:
                decoded_old = wm_decoder(Minus112ZeroOne(imgs_marked)) 
            lossw_old = params.lambda_w * loss_w_1(decoded_old, watermark)
            loss = lossi_old + lossw_old
            loss.backward()

            for name, param in stu.named_parameters():
                if param.grad is not None:
                    accumulated_grads_stu[name] += param.grad.clone()
            for name, param in wm_decoder.named_parameters():
                if param.grad is not None:
                    accumulated_grads_de[name] += param.grad.clone()

            optimizer.base_optimizer.zero_grad()
            imgs_marked = stu.decode(imgs_z, watermark) # b z h/f w/f -> b c h w
            if params.if_tanh:
                decoded = wm_decoder((imgs_marked))
            else:
                decoded = wm_decoder(Minus112ZeroOne(imgs_marked)) 

            lossw = params.lambda_w * loss_w_1(decoded, watermark)
            lossw.backward()
            optimizer.first_step(zero_grad=True)

            # 计算水印损失 + 复原参数
            disable_running_stats(stu)  # <- this is the important line
            disable_running_stats(wm_decoder)

            imgs_marked = stu.decode(imgs_z, watermark) # b z h/f w/f -> b c h w
            if params.if_tanh:
                decoded = wm_decoder((imgs_marked))
            else:
                decoded = wm_decoder(Minus112ZeroOne(imgs_marked)) 

            lossw = params.lambda_w * loss_w_1(decoded, watermark)
            lossw.backward()

            for name, param in stu.named_parameters():
                if param.grad is not None:
                    accumulated_grads_stu[name] += param.grad.clone()
            for name, param in wm_decoder.named_parameters():
                if param.grad is not None:
                    accumulated_grads_de[name] += param.grad.clone()

            optimizer.base_optimizer.zero_grad()

            with torch.no_grad():
                for name, param in stu.named_parameters():
                    param.copy_(cloned_state_dict_stu[name])
                for name, param in wm_decoder.named_parameters():
                    param.copy_(cloned_state_dict_de[name])


            if total_steps % params.log_freq == 0:
                last_time = time.time() - start_time

                wm_predicted = (decoded > 0.0).float()
                bitwise_acc = 100 * (1.0 - torch.mean(torch.abs(watermark - wm_predicted)))

                log = f"{dt_string} {total_steps:06} | LPIPS: {lossi_old.item():.5f} | WM: {lossw.item():.5f} | Acc: {bitwise_acc.item():.2f} | Time:[{format_time(last_time)}]"
                print(log)
                with open(os.path.join(exp_path,'logs.txt'), 'a', encoding='utf-8') as f:
                    f.write(log)
                    f.write('\n')
            
                img_ret.append(lossi_old.item())
                wm_ret.append(bitwise_acc.item())


            # save images during training
            if total_steps % params.save_img_freq == 0:
                save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs),0,1), os.path.join(sample_dir, f'{total_steps:06}_train_orig.png'), nrow=8)
                save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_marked),0,1), os.path.join(sample_dir, f'{total_steps:06}_train_w.png'), nrow=8)
        
            if total_steps % params.save_model_freq == 0:
                save_dict = {
                    'ldm_decoder': stu.state_dict(),
                    'wm_decoder': wm_decoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'params': params,
                }

                # Save checkpoint
                torch.save(save_dict, os.path.join(model_save_dir, f"checkpoint_{total_steps:06d}.pth"))

            if total_steps % params.save_img_freq == 0 and total_steps > params.save_img_freq:
                log = f"{total_steps} Mean Img: {np.mean(img_ret[-20:]):.5f} Mean Acc: {np.mean(wm_ret[-20:]):.5f}"
                with open(os.path.join(exp_path,'results.txt'), 'a', encoding='utf-8') as f:
                    f.write(log)
                    f.write('\n')

        for name, param in stu.named_parameters():
            if param.grad is not None:
                accumulated_grads_stu_history[name] = 0.9 * accumulated_grads_stu_history[name] + 0.1 * accumulated_grads_stu[name]
        for name, param in wm_decoder.named_parameters():
            if param.grad is not None:
                accumulated_grads_de_history[name] = 0.9 * accumulated_grads_de_history[name] + 0.1 * accumulated_grads_de[name]

        for name, param in stu.named_parameters():
            if param.grad is not None:
                param.grad = accumulated_grads_stu_history[name]
        for name, param in wm_decoder.named_parameters():
            if param.grad is not None:
                param.grad = accumulated_grads_de_history[name]

        optimizer.base_optimizer.step()
        optimizer.base_optimizer.zero_grad()

if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    main(params)