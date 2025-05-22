import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
import SimpleITK as sitk
import os

def save_combine_oar(tmp_dict, need_list, save_dir, OAR_DICT, spacing):
    comb_oar = np.zeros(tmp_dict['img'].shape)
    for key in OAR_DICT.keys():
        if key not in need_list:
            continue
        if key in tmp_dict.keys():
            single_oar_mask = tmp_dict[key].round()
            single_oar_psdm = PSDM_transform(single_oar_mask, spacing)
        else:
            single_oar_mask = np.zeros(tmp_dict['img'].shape)
            single_oar_psdm = np.zeros(tmp_dict['img'].shape)
        save_npy2nii(single_oar_mask.astype(np.uint8), spacing,
                     os.path.join(save_dir, 'original_mask', '{}.nii.gz'.format(key)))
        save_npy2nii(single_oar_psdm, spacing,
                     os.path.join(save_dir, 'PSDM', '{}.nii.gz'.format(key)))
        comb_oar = np.maximum(comb_oar, single_oar_mask * OAR_DICT[key])
    save_npy2nii(comb_oar.astype(np.uint8), spacing, os.path.join(save_dir, 'OAR_masks.nii.gz'))

def save_combine_ptv(tmp_dict, scaled_dose_dict, save_dir, spacing, save_name):
    comb_ptv = np.zeros(tmp_dict['img'].shape)
    for key in scaled_dose_dict.keys():
        tmp_ptv_mask = tmp_dict[key]
        tmp_ptv_psdm = PSDM_transform(tmp_ptv_mask, spacing)
        save_npy2nii(tmp_ptv_mask.astype(np.uint8), spacing,
                     os.path.join(save_dir, 'original_mask', '{}.nii.gz'.format(key)))
        save_npy2nii(tmp_ptv_psdm, spacing,
                     os.path.join(save_dir, 'PSDM', '{}.nii.gz'.format(key)))
        comb_ptv = np.maximum(comb_ptv, tmp_ptv_mask * scaled_dose_dict[key])
    save_npy2nii(comb_ptv, spacing, os.path.join(save_dir, '{}.nii.gz'.format(save_name)))

def poly_lr_scheduler(optimizer, base_lr, n_iter, lr_decay_iter=1, max_iter=100, power=0.9):
    if n_iter % lr_decay_iter == 0 and n_iter <= max_iter:
        lr = base_lr * (1 - n_iter / max_iter) ** power
        for param_gourp in optimizer.param_groups:
            param_gourp['lr'] = lr

def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def clip_norm(img_, clip_range=(-1000, 1000), norm_range=(-1, 1)):
    img_normed = (np.clip(img_, clip_range[0], clip_range[1]) - clip_range[0]) / (clip_range[1] - clip_range[0])
    img_normed = img_normed * (norm_range[1] - norm_range[0]) + norm_range[0]
    return img_normed

def save_npy2nii(img, spacing, save_path):
    img_nii = sitk.GetImageFromArray(img)
    img_nii.SetSpacing([spacing[1], spacing[2], spacing[0]])
    sitk.WriteImage(img_nii, save_path)

def PSDM_transform(mask, spacing):
    dis_map_p = ndimage.morphology.distance_transform_edt(mask, sampling=spacing)
    dis_map_n = ndimage.morphology.distance_transform_edt(1 - mask, sampling=spacing)
    dis_map = (dis_map_p - dis_map_n) / 100
    return dis_map

def get_combine_oar_mask(tmp_dict, need_list, OAR_DICT):
    comb_oar = np.zeros(tmp_dict['img'].shape)
    for key in OAR_DICT.keys():
        if key not in need_list:
            continue
        if key in tmp_dict.keys():
            single_oar_mask = tmp_dict[key].round()
        else:
            single_oar_mask = np.zeros(tmp_dict['img'].shape)
        comb_oar = np.maximum(comb_oar, single_oar_mask * OAR_DICT[key])
    return comb_oar

def get_combine_ptv_mask(tmp_dict, scaled_dose_dict):
    comb_ptv = np.zeros(tmp_dict['img'].shape)
    for key in scaled_dose_dict.keys():
        tmp_ptv_mask = tmp_dict[key]
        comb_ptv = np.maximum(comb_ptv, tmp_ptv_mask * scaled_dose_dict[key])
    return comb_ptv