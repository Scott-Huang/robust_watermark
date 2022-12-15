import os, sys
if not os.path.exists('./model'):
    print('call this module in the src directory')
    exit()

import torch
import cv2 as cv
import numpy as np
from PIL import Image
from args_util import AOT_Args, DeepFillv2_Args

def aot_gan():
    import importlib
    from torchvision.transforms import ToTensor
    sys.path.append('./model/AOT-GAN-for-Inpainting/src')
    args = AOT_Args()
    args.pre_train = './model/AOT-GAN-for-Inpainting/pre_trained/G0000000.pt'
    net = importlib.import_module('model.'+args.model)
    model = net.InpaintGenerator(args).cuda()
    model.load_state_dict(torch.load(args.pre_train, map_location='cuda'))
    model.eval()
    def inpaint(img, mask):
        if img.isinstance(Image.Image):
            img = img.convert('RGB')
        elif img.isinstance(np.ndarray) and img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        img = ToTensor()(img)
        mask = ToTensor()(mask).unsqueeze(0)
        image, mask = image.cuda(), mask.cuda()
        image_masked = image * (1 - mask.float()) + mask
        with torch.no_grad():
            pred_img = model(image_masked, mask)
        comp_imgs = (1 - mask) * image + mask * pred_img
        comp_imgs = comp_imgs.cpu().numpy()
        return comp_imgs
    return inpaint

def deepfill():
    sys.path.insert(0,'./model/DeepFillv2_Pytorch')
    from utils import create_generator
    opt = DeepFillv2_Args()
    model_name = 'deepfillv2_WGAN_G_epoch40_batchsize4.pth'
    model_name = os.path.join('model/DeepFillv2_Pytorch/pretrained_model', model_name)
    pretrained_dict = torch.load(model_name)
    generator = create_generator(opt).eval()
    generator.load_state_dict(pretrained_dict)
    generator = generator.cuda()
    def inpaint(img, mask):
        if img.isinstance(Image.Image):
            img = img.convert('RGB')
        elif img.isinstance(np.ndarray) and img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        img = img.cuda()
        mask = mask.cuda()
        with torch.no_grad():
            _, second_out = generator(img, mask)
        second_out_wholeimg = img * (1 - mask) + second_out * mask
        return second_out_wholeimg.cpu().numpy()
    return inpaint

def stable_diffusion():
    from diffusers import StableDiffusionInpaintPipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")
    def inpaint(img, mask):
        if img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        size = img.shape[:2]
        img = Image.fromarray(img).resize((512, 512))
        mask = Image.fromarray(mask).resize((512, 512))
        prompt = "" # no prompt
        return np.asarray(pipe(prompt=prompt, image=img, mask_image=mask).images[0].resize((size[1], size[0])))
    return inpaint
