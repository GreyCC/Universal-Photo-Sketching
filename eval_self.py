import os
import argparse
from datasets import is_image_file
from network import *
import torch.backends.cudnn as cudnn
from os.path import *
from os import listdir
import torchvision.utils as vutils
from PIL import Image, ImageOps
from img_utils import modcrop, rescale_img_scale
from torchvision.transforms import functional as TF
import time

parser = argparse.ArgumentParser()
parser.add_argument('--P2S_dir', type=str, default='models/P2S_base.pth')
parser.add_argument("--image_dataset", default="Test/", help='image dataset')

################# PREPARATIONS #################
opt = parser.parse_args()

device = torch.device("cuda:0")
cudnn.benchmark = True

################# MODEL #################
P2S = P2Sv2()

if os.path.exists(opt.P2S_dir):
    pretrained_dict = torch.load(opt.P2S_dir, map_location=lambda storage, loc: storage)
    model_dict = P2S.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    P2S.load_state_dict(model_dict)
    print('pretrained Photo2Sketch model is loaded!')

################# GPU  #################
P2S.to(device)

################# Testing #################
def eval():
    P2S.eval()

    HR_filename = os.path.join(opt.image_dataset, 'test_img')
    SR_filename = os.path.join(opt.image_dataset, 'sketch_result')

    gt_image = [join(HR_filename, x) for x in listdir(HR_filename) if is_image_file(x)]
    output_image = [join(SR_filename, x) for x in listdir(HR_filename) if is_image_file(x)]

    for i in range(gt_image.__len__()):
        HR = Image.open(gt_image[i]).convert('RGB')

        ############# Initial autocontrast for better sketching #################
        HR = ImageOps.autocontrast(HR, cutoff=(0.2, 4), preserve_tone=True)

        ############# Adjust input size to control number of edge #################
        HR = transform(HR, 256)

        with torch.no_grad():
            img = HR.unsqueeze(0).to(device)
            pro_time = time.time()
            out = P2S(img)
            pro_time = time.time() - pro_time
            print("Eval time: ", pro_time)
        torch.cuda.empty_cache()
        out = out.repeat(1, 3, 1, 1).clamp(0, 1).cpu().data
        vutils.save_image(out, f'{output_image[i][:-4]}_sketch.png', normalize=True, scale_each=True, nrow=2)


def transform(image, size):
    # Resize
    H, W = image.size
    if H > W: #vertical image
        scale = size / W
    else: #Horizontal image
        scale = size / H
    image = rescale_img_scale(image, scale)

    image = modcrop(image, 8)
    image = TF.to_tensor(image)
    return image

eval()
