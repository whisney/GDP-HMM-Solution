import numpy as np
import os
import torch
from Nii_utils import NiiDataRead
from monai.transforms import Compose, RandSpatialCrop, SpatialPad, RandRotate, RandZoom, Rand3DElastic, RandFlip, ToTensor
from torch.utils.data import DataLoader, Dataset
import pickle
from skimage.morphology import dilation, ball
import matplotlib.pyplot as plt
import pandas as pd
from utils import clip_norm

class Dataset_Dose(Dataset):
    def __init__(self, data_dir='data', split_file='data/all_split_5fold_seed0.pkl', fold=0, subset='train',
                 aug=True, norm_mode=1, input_mode=1, aug_mode=1, size=(64, 256, 256)):
        self.data_dir = data_dir
        self.size = size
        pkl_data = pickle.load(open(split_file, 'rb'))
        if fold == 5:
            if subset == 'train':
                self.ID_list_all = pkl_data[0]['train'] + pkl_data[0]['val']
            elif subset == 'val':
                self.ID_list_all = pkl_data[0]['val'][:16]
        else:
            self.ID_list_all = pkl_data[fold][subset]
        self.input_mode = input_mode
        self.len = len(self.ID_list_all)

        if norm_mode == 1:
            self.CT_clip = (-1000, 1500)
            self.dose_clip = (0, 75)
            self.norm_range = (-1, 1)
            self.oar_div_factor = 30.
            self.beam_div_factor = 90.

        if aug_mode == 1:
            if aug:
                self.transforms = Compose([
                    RandSpatialCrop(roi_size=size, random_center=True, random_size=False),
                    SpatialPad(spatial_size=size, method="symmetric", mode="constant"),
                    RandRotate(range_x=0.0, range_y=0.0, range_z=np.pi / 360 * 50, prob=0.3, keep_size=True,
                               mode="nearest", padding_mode="zeros"),
                    RandZoom(min_zoom=0.8, max_zoom=1.2, mode="nearest", padding_mode="constant", prob=0.3, keep_size=True),
                    RandFlip(spatial_axis=1, prob=0.3),
                    RandFlip(spatial_axis=2, prob=0.3),
                    ToTensor()
                ])
            else:
                self.transforms = Compose([RandSpatialCrop(roi_size=size, random_center=True, random_size=False),
                                           SpatialPad(spatial_size=size, method="symmetric", mode="constant"),
                                           ToTensor()])
        elif aug_mode == 2:
            if aug:
                self.transforms = Compose([
                    RandSpatialCrop(roi_size=size, random_center=True, random_size=False),
                    SpatialPad(spatial_size=size, method="symmetric", mode="constant"),
                    RandRotate(range_x=0.0, range_y=0.0, range_z=np.pi / 360 * 70, prob=0.5, keep_size=True,
                               mode="nearest", padding_mode="zeros"),
                    RandZoom(min_zoom=0.6, max_zoom=1.4, mode="nearest", padding_mode="constant", prob=0.5, keep_size=True),
                    RandFlip(spatial_axis=1, prob=0.5),
                    RandFlip(spatial_axis=2, prob=0.5),
                    ToTensor()
                ])
            else:
                self.transforms = Compose([RandSpatialCrop(roi_size=size, random_center=True, random_size=False),
                                           SpatialPad(spatial_size=size, method="symmetric", mode="constant"),
                                           ToTensor()])

    def __getitem__(self, idx):
        ID = self.ID_list_all[idx]
        CT, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'img.nii.gz'))
        input_img = [clip_norm(CT, self.CT_clip, self.norm_range)[np.newaxis, ...]]
        site = ID.split('/')[0].split('_')[1]

        if self.input_mode == 1:  # input_channels = 9
            input_one, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'OAR_masks.nii.gz'))
            input_img.append(input_one[np.newaxis, ...] / self.oar_div_factor)
            input_one, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'comb_optptv.nii.gz'))
            input_img.append(clip_norm(input_one, self.dose_clip, self.norm_range)[np.newaxis, ...])
            input_one, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'comb_ptv.nii.gz'))
            input_img.append(clip_norm(input_one, self.dose_clip, self.norm_range)[np.newaxis, ...])
            input_one, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'PSDM', 'isocenter.nii.gz'))
            input_img.append(input_one[np.newaxis, ...])
            input_one, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'original_mask', 'angle_plate.nii.gz'))
            input_img.append(input_one[np.newaxis, ...])
            input_one, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'PSDM', 'angle_plate.nii.gz'))
            input_img.append(input_one[np.newaxis, ...])
            input_one, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'original_mask', 'beam_plate.nii.gz'))
            input_img.append(input_one[np.newaxis, ...] / self.beam_div_factor)
            input_one, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'original_mask', 'Body.nii.gz'))
            input_img.append(input_one[np.newaxis, ...])
            cylinder_kernel = np.zeros((3, 11, 11), dtype=np.uint8)
            cylinder_kernel[1] = ball(5)[5]
            mask_for_loss = dilation(input_one, cylinder_kernel)
            input_img.append(mask_for_loss[np.newaxis, ...])

        dose, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'dose.nii.gz'))
        input_img.append(clip_norm(dose, self.dose_clip, self.norm_range)[np.newaxis, ...])
        input_img = np.concatenate(input_img, axis=0)
        img_augmented = self.transforms(input_img).as_tensor()
        input_img = img_augmented[:-2]
        mask_for_loss = img_augmented[-2].unsqueeze(0)
        dose = img_augmented[-1].unsqueeze(0)
        isVMAT = pd.read_csv(os.path.join(self.data_dir, ID, 'isVMAT.csv'))['isVMAT'][0]
        if (not isVMAT) and (site == 'HaN'):
            isVMAT = torch.tensor(0)
        elif isVMAT and (site == 'HaN'):
            isVMAT = torch.tensor(1)
        elif (not isVMAT) and (site == 'Lung'):
            isVMAT = torch.tensor(2)
        elif isVMAT and (site == 'Lung'):
            isVMAT = torch.tensor(3)
        return input_img, dose, isVMAT, mask_for_loss

    def __len__(self):
        return self.len

if __name__ == '__main__':
    train_data = Dataset_Dose(data_dir=r'NII_data',
                              split_file=r'NII_data/all_split_5fold_seed0.pkl', fold=0,
                              subset='train', aug=True, norm_mode=1, input_mode=1, aug_mode=1, size=(128, 128, 192))
    train_dataloader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    for i, (input_img, dose, isVMAT, mask_for_loss) in enumerate(train_dataloader):
        print(input_img.shape, dose.shape, mask_for_loss.shape)
        print(isVMAT)
        input_img = input_img.squeeze(0).numpy()
        dose = dose.squeeze(0).squeeze(0).numpy()
        mask_for_loss = mask_for_loss.squeeze(0).squeeze(0).numpy()
        plt.subplot(221)
        plt.imshow(input_img[0, 32])
        plt.subplot(222)
        plt.imshow(input_img[4, 32])
        plt.subplot(223)
        plt.imshow(dose[32])
        plt.subplot(224)
        plt.imshow(mask_for_loss[32])
        plt.show()