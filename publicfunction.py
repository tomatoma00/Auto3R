import shutil
import subprocess
import torch
import os
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import convert_image_dtype
import torchvision.io as tvio
from hyperiqa import HyperNet,TargetNet

####################################################
# Note: Due to the size limitation of the supplementary materials, 
# we only provide pre-trained UQ for rendered images, 
# and the UQ model for depth maps will be replaced by the hyperiqa
#####################################################


model_hyper = HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
model_hyper.train(False)
# pretrained/koniq_pretrained.pkl can be download from the 
# hyperiqa (Su, Shaolin et.al. CVPR 2020)
model_hyper.load_state_dict((torch.load('pretrained/koniq_pretrained.pkl')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxerror = []
averageerror = []
    
# get best view based on Uq
@torch.no_grad()
def qualitycheck(model,images,depths,out_dir,namelist,batch_size = 32,iter = -1):
    B, C, H, W = images.shape
    if H == W == 512:                 # nothing to do
        pass
    else:
        edge = max(H, W)
        pad_h, pad_w = edge - H, edge - W
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        images = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom), value=0)
        depths = F.pad(depths, (pad_left, pad_right, pad_top, pad_bottom), value=0)
        images = F.interpolate(images, size=(512, 512), mode='bilinear', align_corners=False)
        depths = F.interpolate(depths, size=(512, 512), mode='bilinear', align_corners=False)

    depths2 = torch.nn.functional.pad(depths, (3,3,3,3), mode="constant", value=0).cpu()
    depths = depths.repeat(1, 3, 1, 1) 
    r_slice, c_slice = orig_slice(H, W, pad_left, pad_right, pad_top, pad_bottom)
    predsimg,predsdepth = [],[]
    for i in range(0, images.size(0), batch_size):
        x1 = images[i:i+batch_size].to(device)
        x2 = depths[i:i+batch_size].to(device)
        outimg = model(x1).cpu()
        x2 = F.interpolate(x2, size=(224, 224),
                  mode='bilinear',
                  align_corners=False) 
        
        outdepthpara = model_hyper(x2)
        outdepthmodel = TargetNet(outdepthpara)

        for param in outdepthmodel.parameters():
            param.requires_grad = False
        outdepth = outdepthmodel(outdepthpara['target_in_vec']).view(-1)/100.0
        predsimg.append(outimg)
        predsdepth.append(outdepth)

    predsimg = torch.cat(predsimg, dim=0)                    # B×1×518×518
    predsimg = torch.clamp(predsimg, 0.0, 1.0)#.squeeze(1)
    predsdepth = torch.cat(predsdepth, dim=0).cpu()
    predsdepth = 1-torch.clamp(predsdepth, 0.0, 1.0).view(B, 1, 1, 1) 
    if iter==-1:
        predsimg=0.5*predsdepth+predsimg+predsimg*predsdepth
    else:
        predsimg=predsimg*(depths2)+0.5*predsimg*(1-predsdepth)*(depths2)+(iter/40000)*predsdepth + 0.8*predsimg
    predsimg = predsimg.squeeze(1)
    sum_per_img = predsimg[...,r_slice, c_slice].sum(dim=(1, 2)) 
    max_idx = sum_per_img.argmax()
    print("max_uq",sum_per_img[max_idx]/(518*518))
    return max_idx,sum_per_img[max_idx]/(518*518)

def robust_normalize_gpu(x: torch.Tensor, trim=0.1):
    flat = x.view(-1)
    n = flat.numel()
    k_low  = int(n * trim + 0.5)
    k_high = n - 2*k_low - 1 # 75-78
    low  = torch.kthvalue(flat, k_low + 1)[0]
    high = torch.kthvalue(flat, k_high + 1)[0]

    out = ((x - low) / (high - low + 1e-8)).clamp_(0, 1)   # normalize
    return (out+0.5)/1.5                  # keep zeros
def orig_slice(h, w, pl, pr, pt, pb, H=512, W=512):
    """return row/col slices of the original image inside the 512×512 map"""
    return (slice(int(pt * H / (h + pt + pb)), int((h + pt) * H / (h + pt + pb))),
            slice(int(pl * W / (w + pl + pr)), int((w + pl) * W / (w + pl + pr))))
