#! /usr/bin/env python3
import os
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
from monai.data import Dataset, DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImage,
    LoadImaged,
    NormalizeIntensityd,
    Resized,
    SaveImaged,
    SplitDimd
)
import torch
sys.path.append('../DenseNetFCN3D-pytorch')
from densenet import *

'''
Run inference on images that have been organise correctly:
DWI and ADC images should be stack into a two-channel NIFTI
organise_data.py code does this
DenseNet code found at https://github.com/FredaWerdiger/DenseNetFCN3D-pytorch

'''

def define_dvalues(dwi_img):
    steps = int(dwi_img.shape[2] / 18)
    rem = int(dwi_img.shape[2] / steps) - 18

    if rem == 0:
        d_min = 0
        d_max = dwi_img.shape[2]
    elif rem % 2 == 0:
        d_min = 0 + int(rem / 2 * steps) + 1
        d_max = dwi_img.shape[2] - int(rem / 2 * steps) + 1

    elif rem % 2 != 0:
        d_min = 0 + math.ceil(rem * steps / 2)
        d_max = dwi_img.shape[2] - math.ceil(rem / 2 * steps) + 1

    d = range(d_min, d_max, steps)

    if len(d) == 19:
        d = d[1:]
    return d

def create_mrlesion_img(dwi_img, dwi_lesion_img, savefile, d, ext='png', dpi=250):
    dwi_lesion_img = np.rot90(dwi_lesion_img)
    dwi_img = np.rot90(dwi_img)

    mask = dwi_lesion_img < 1
    masked_im = np.ma.array(dwi_img, mask=~mask)

    fig, axs = plt.subplots(3, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.6, wspace=-0.1)
    axs = axs.ravel()

    for i in range(len(d)):
        axs[i].imshow(dwi_img[:, :, d[i]], cmap='gray', interpolation='hanning', vmin=0, vmax=300)
        axs[i].imshow(dwi_lesion_img[:, :, d[i]], cmap='Reds', interpolation='hanning', alpha=0.5, vmin=-2, vmax=1)
        axs[i].imshow(masked_im[:, :, d[i]], cmap='gray', interpolation='hanning', alpha=1, vmin=0, vmax=300)
        axs[i].axis('off')
    plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
    plt.close()


def main(path_to_images):

    test_transforms = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Resized(keys="image",
                    mode='trilinear',
                    align_corners=True,
                    spatial_size=(128, 128, 128)),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys="image")]
    )

    test_files = [{"image": image_name} for image_name in glob.glob(os.path.join(path_to_images, 'images/*'))]

    out_path = os.path.join(path_to_images, 'pred')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    test_ds = Dataset(
        data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=1)

    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        EnsureChannelFirstd(keys="label"),
        Invertd(
            keys="pred",
            transform=test_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        SplitDimd(keys="pred", dim=0, keepdim=True,
                  output_postfixes=['inverted', 'good']),
        SaveImaged(
            keys="pred_good",
            meta_keys="pred_meta_dict",
            output_dir=out_path,
            output_postfix="pred",
            resample=False,
            separate_folder=False)
    ])

    loader = LoadImage(image_only=False)
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    model = DenseNetFCN(
      ch_in=2,
       ch_out_init=48,
       num_classes=2,
       growth_rate=16,
       layers=(4, 5, 7, 10, 12),
       bottleneck=True,
       bottleneck_layer=15
    ).to(device)

    model.load_state_dict(torch.load('dwi_densenet.pth'))
    model.eval()

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_inputs = test_data["image"].to(device)
            test_data["pred"] = model(test_inputs)

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]
            test_output,test_image = from_engine(["pred", "image"])(test_data)

            original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])
            original_image = original_image[0]  # image data
            original_image = original_image[:, :, :, 0]
            prediction = test_output[0][1].detach().numpy()
            subject = test_data[0]["image_meta_dict"]["filename_or_obj"].split('.nii.gz')[0]
            save_loc = os.path.join(out_path, subject + '_pred.png')

            create_mrlesion_img(
                original_image,
                prediction,
                save_loc,
                define_dvalues(original_image),
                'png',
                dpi=300)

if __name__ == '__main__':
    path_to_images = sys.argv[1]
    main(path_to_images)
