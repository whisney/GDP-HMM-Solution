import numpy as np
import os
import json
from skimage.morphology import ball, binary_dilation
from utils import PSDM_transform, save_npy2nii, save_combine_oar, save_combine_ptv
from scipy import ndimage
import pandas as pd
from multiprocessing import Pool

scale_dose_Dict = json.load(open(r'train_data/meta_files/PTV_DICT.json', 'r'))
pat_obj_dict = json.load(open(r'train_data/meta_files/Pat_Obj_DICT.json', 'r'))

def npy2nii_processing_HaN(full_name):
    HaN_OAR_LIST = ['Cochlea_L', 'Cochlea_R', 'Eyes', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R', 'Chiasim',
                    'LacrimalGlands', 'BrachialPlexus', 'Brain', 'BrainStem_03', 'Esophagus', 'Lips', 'Lungs',
                    'Trachea',
                    'Posterior_Neck', 'Shoulders', 'Larynx-PTV', 'Mandible-PTV', 'OCavity-PTV', 'ParotidCon-PTV',
                    'Parotidlps-PTV', 'Parotids-PTV', 'PharConst-PTV', 'Submand-PTV', 'SubmandL-PTV', 'SubmandR-PTV',
                    'Thyroid-PTV', 'SpinalCord_05']
    HaN_OAR_DICT = {HaN_OAR_LIST[i]: (i + 1) for i in range(len(HaN_OAR_LIST))}
    OAR_LIST = HaN_OAR_LIST
    OAR_DICT = HaN_OAR_DICT
    data_dir = r'train_data/train_HaN/train'
    save_dir = r'NII_data/train_HaN'

    print(full_name)
    ID = full_name.replace('.npz', '')
    PatientID = ID.split('+')[0]
    if len(str(PatientID)) < 3:
        PatientID = f"{PatientID:0>3}"

    os.makedirs(os.path.join(save_dir, ID), exist_ok=True)
    os.makedirs(os.path.join(save_dir, ID, 'original_mask'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, ID, 'PSDM'), exist_ok=True)

    data_npz = np.load(os.path.join(data_dir, full_name), allow_pickle=True)
    data_npz = dict(data_npz)['arr_0'].item()

    img_shape = data_npz['img'].shape

    # CT image
    save_npy2nii(data_npz['img'], [2.0, 2.5, 2.5], os.path.join(save_dir, ID, 'img.nii.gz'))

    # Dose
    ptv_highdose = scale_dose_Dict[PatientID]['PTV_High']['PDose']
    dose_nii = data_npz['dose'] * data_npz['dose_scale']
    PTVHighOPT = scale_dose_Dict[PatientID]['PTV_High']['OPTName']
    norm_scale = ptv_highdose / (np.percentile(dose_nii[data_npz[PTVHighOPT].astype('bool')], 3) + 1e-5)
    dose_nii = dose_nii * norm_scale
    save_npy2nii(dose_nii, [2.0, 2.5, 2.5], os.path.join(save_dir, ID, 'dose.nii.gz'))

    # isocenter
    isocenter = data_npz['isocenter']
    isocenter_mask_nii = np.zeros(data_npz['img'].shape)
    isocenter_mask_nii[int(isocenter[0]), int(isocenter[1]), int(isocenter[2])] = 1
    structuring_element = ball(3)
    isocenter_mask_nii = binary_dilation(isocenter_mask_nii, structuring_element)
    isocenter_psdm_nii = PSDM_transform(mask=isocenter_mask_nii, spacing=[2.0, 2.5, 2.5])
    save_npy2nii(isocenter_mask_nii.astype(np.uint8), [2.0, 2.5, 2.5],
                 os.path.join(save_dir, ID, 'original_mask', 'isocenter.nii.gz'))
    save_npy2nii(isocenter_psdm_nii, [2.0, 2.5, 2.5],
                 os.path.join(save_dir, ID, 'PSDM', 'isocenter.nii.gz'))

    # angle_plate
    angle_plate_mask_nii = np.zeros(img_shape)
    z_begin = int(isocenter[0]) - 5
    z_end = int(isocenter[0]) + 5
    z_begin = max(0, z_begin)
    z_end = min(angle_plate_mask_nii.shape[0], z_end)
    D3_plate = np.repeat(data_npz['angle_plate'][np.newaxis, :, :], z_end - z_begin, axis=0)
    if D3_plate.shape[1] != angle_plate_mask_nii.shape[1] or D3_plate.shape[2] != angle_plate_mask_nii.shape[2]:
        D3_plate = ndimage.zoom(D3_plate, (
        1, angle_plate_mask_nii.shape[1] / D3_plate.shape[1], angle_plate_mask_nii.shape[2] / D3_plate.shape[2]), order=0)
    angle_plate_mask_nii[z_begin: z_end] = D3_plate
    angle_plate_psdm_nii = PSDM_transform(mask=angle_plate_mask_nii, spacing=[2.0, 2.5, 2.5])
    save_npy2nii(angle_plate_mask_nii.astype(np.uint8), [2.0, 2.5, 2.5],
                 os.path.join(save_dir, ID, 'original_mask', 'angle_plate.nii.gz'))
    save_npy2nii(angle_plate_psdm_nii, [2.0, 2.5, 2.5],
                 os.path.join(save_dir, ID, 'PSDM', 'angle_plate.nii.gz'))

    # beam_plate
    beam_plate_nii = data_npz['beam_plate']
    save_npy2nii(beam_plate_nii.astype(np.uint8), [2.0, 2.5, 2.5],
                 os.path.join(save_dir, ID, 'original_mask', 'beam_plate.nii.gz'))

    # OAR
    try:
        need_list = pat_obj_dict[ID.split('+')[0]]
    except:
        need_list = OAR_LIST
        print(ID.split('+')[0], '-------------not in the pat_obj_dict')
    save_combine_oar(data_npz, need_list, os.path.join(save_dir, ID), OAR_DICT, [2.0, 2.5, 2.5])

    # PTV
    opt_dose_dict = {}
    dose_dict = {}
    for key in scale_dose_Dict[PatientID].keys():
        if key in ['PTV_High', 'PTV_Mid', 'PTV_Low']:
            opt_dose_dict[scale_dose_Dict[PatientID][key]['OPTName']] = scale_dose_Dict[PatientID][key]['PDose']
            if key != 'PTV_High':
                dose_dict[scale_dose_Dict[PatientID][key]['StructName']] = scale_dose_Dict[PatientID][key]['PDose']
            else:
                dose_dict[scale_dose_Dict[PatientID][key]['OPTName']] = scale_dose_Dict[PatientID][key]['PDose']
    save_combine_ptv(data_npz, opt_dose_dict, os.path.join(save_dir, ID), [2.0, 2.5, 2.5], 'comb_optptv')
    save_combine_ptv(data_npz, dose_dict, os.path.join(save_dir, ID), [2.0, 2.5, 2.5], 'comb_ptv')
    opt_dose_df = pd.DataFrame({k: [opt_dose_dict[k]] for k in opt_dose_dict.keys()})
    opt_dose_df.to_csv(os.path.join(save_dir, ID, 'opt_dose_dict.csv'), index=False)
    dose_df = pd.DataFrame({k: [dose_dict[k]] for k in dose_dict.keys()})
    dose_df.to_csv(os.path.join(save_dir, ID, 'dose_dict.csv'), index=False)

    # Body
    Body_mask = data_npz['Body']
    Body_psdm = PSDM_transform(mask=Body_mask, spacing=[2.0, 2.5, 2.5])
    save_npy2nii(Body_mask.astype(np.uint8), [2.0, 2.5, 2.5],
                 os.path.join(save_dir, ID, 'original_mask', 'Body.nii.gz'))
    save_npy2nii(Body_psdm, [2.0, 2.5, 2.5],
                 os.path.join(save_dir, ID, 'PSDM', 'Body.nii.gz'))

    # isVMAT
    isVMAT_df = pd.DataFrame({'isVMAT': [data_npz['isVMAT']]})
    isVMAT_df.to_csv(os.path.join(save_dir, ID, 'isVMAT.csv'), index=False)
    return full_name

def npy2nii_processing_Lung(full_name):
    Lung_OAR_LIST = ["PTV_Ring.3-2", "Total Lung-GTV", "SpinalCord", "Heart", "LAD", "Esophagus", "BrachialPlexus",
                     "GreatVessels", "Trachea", "Body_Ring0-3"]
    Lung_OAR_DICT = {Lung_OAR_LIST[i]: (i + 1) for i in range(len(Lung_OAR_LIST))}
    OAR_LIST = Lung_OAR_LIST
    OAR_DICT = Lung_OAR_DICT
    data_dir = r'train_data/train_Lung/train'
    save_dir = r'NII_data/train_Lung'

    print(full_name)
    ID = full_name.replace('.npz', '')
    PatientID = ID.split('+')[0]
    if len(str(PatientID)) < 3:
        PatientID = f"{PatientID:0>3}"

    os.makedirs(os.path.join(save_dir, ID), exist_ok=True)
    os.makedirs(os.path.join(save_dir, ID, 'original_mask'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, ID, 'PSDM'), exist_ok=True)

    data_npz = np.load(os.path.join(data_dir, full_name), allow_pickle=True)
    data_npz = dict(data_npz)['arr_0'].item()

    img_shape = data_npz['img'].shape

    # CT image
    save_npy2nii(data_npz['img'], [2.0, 2.5, 2.5], os.path.join(save_dir, ID, 'img.nii.gz'))

    # Dose
    ptv_highdose = scale_dose_Dict[PatientID]['PTV_High']['PDose']
    dose_nii = data_npz['dose'] * data_npz['dose_scale']
    PTVHighOPT = scale_dose_Dict[PatientID]['PTV_High']['OPTName']
    norm_scale = ptv_highdose / (np.percentile(dose_nii[data_npz[PTVHighOPT].astype('bool')], 3) + 1e-5)
    dose_nii = dose_nii * norm_scale
    save_npy2nii(dose_nii, [2.0, 2.5, 2.5], os.path.join(save_dir, ID, 'dose.nii.gz'))

    # isocenter
    isocenter = data_npz['isocenter']
    isocenter_mask_nii = np.zeros(data_npz['img'].shape)
    isocenter_mask_nii[int(isocenter[0]), int(isocenter[1]), int(isocenter[2])] = 1
    structuring_element = ball(3)
    isocenter_mask_nii = binary_dilation(isocenter_mask_nii, structuring_element)
    isocenter_psdm_nii = PSDM_transform(mask=isocenter_mask_nii, spacing=[2.0, 2.5, 2.5])
    save_npy2nii(isocenter_mask_nii.astype(np.uint8), [2.0, 2.5, 2.5],
                 os.path.join(save_dir, ID, 'original_mask', 'isocenter.nii.gz'))
    save_npy2nii(isocenter_psdm_nii, [2.0, 2.5, 2.5],
                 os.path.join(save_dir, ID, 'PSDM', 'isocenter.nii.gz'))

    # angle_plate
    angle_plate_mask_nii = np.zeros(img_shape)
    z_begin = int(isocenter[0]) - 5
    z_end = int(isocenter[0]) + 5
    z_begin = max(0, z_begin)
    z_end = min(angle_plate_mask_nii.shape[0], z_end)
    D3_plate = np.repeat(data_npz['angle_plate'][np.newaxis, :, :], z_end - z_begin, axis=0)
    if D3_plate.shape[1] != angle_plate_mask_nii.shape[1] or D3_plate.shape[2] != angle_plate_mask_nii.shape[2]:
        D3_plate = ndimage.zoom(D3_plate, (
        1, angle_plate_mask_nii.shape[1] / D3_plate.shape[1], angle_plate_mask_nii.shape[2] / D3_plate.shape[2]), order=0)
    angle_plate_mask_nii[z_begin: z_end] = D3_plate
    angle_plate_psdm_nii = PSDM_transform(mask=angle_plate_mask_nii, spacing=[2.0, 2.5, 2.5])
    save_npy2nii(angle_plate_mask_nii.astype(np.uint8), [2.0, 2.5, 2.5],
                 os.path.join(save_dir, ID, 'original_mask', 'angle_plate.nii.gz'))
    save_npy2nii(angle_plate_psdm_nii, [2.0, 2.5, 2.5],
                 os.path.join(save_dir, ID, 'PSDM', 'angle_plate.nii.gz'))

    # beam_plate
    beam_plate_nii = data_npz['beam_plate']
    save_npy2nii(beam_plate_nii.astype(np.uint8), [2.0, 2.5, 2.5],
                 os.path.join(save_dir, ID, 'original_mask', 'beam_plate.nii.gz'))

    # OAR
    try:
        need_list = pat_obj_dict[ID.split('+')[0]]
    except:
        need_list = OAR_LIST
        print(ID.split('+')[0], '-------------not in the pat_obj_dict')
    save_combine_oar(data_npz, need_list, os.path.join(save_dir, ID), OAR_DICT, [2.0, 2.5, 2.5])

    # PTV
    opt_dose_dict = {}
    dose_dict = {}
    for key in scale_dose_Dict[PatientID].keys():
        if key in ['PTV_High', 'PTV_Mid', 'PTV_Low']:
            opt_dose_dict[scale_dose_Dict[PatientID][key]['OPTName']] = scale_dose_Dict[PatientID][key]['PDose']
            if key != 'PTV_High':
                dose_dict[scale_dose_Dict[PatientID][key]['StructName']] = scale_dose_Dict[PatientID][key]['PDose']
            else:
                dose_dict[scale_dose_Dict[PatientID][key]['OPTName']] = scale_dose_Dict[PatientID][key]['PDose']
    save_combine_ptv(data_npz, opt_dose_dict, os.path.join(save_dir, ID), [2.0, 2.5, 2.5], 'comb_optptv')
    save_combine_ptv(data_npz, dose_dict, os.path.join(save_dir, ID), [2.0, 2.5, 2.5], 'comb_ptv')
    opt_dose_df = pd.DataFrame({k: [opt_dose_dict[k]] for k in opt_dose_dict.keys()})
    opt_dose_df.to_csv(os.path.join(save_dir, ID, 'opt_dose_dict.csv'), index=False)
    dose_df = pd.DataFrame({k: [dose_dict[k]] for k in dose_dict.keys()})
    dose_df.to_csv(os.path.join(save_dir, ID, 'dose_dict.csv'), index=False)

    # Body
    Body_mask = data_npz['Body']
    Body_psdm = PSDM_transform(mask=Body_mask, spacing=[2.0, 2.5, 2.5])
    save_npy2nii(Body_mask.astype(np.uint8), [2.0, 2.5, 2.5],
                 os.path.join(save_dir, ID, 'original_mask', 'Body.nii.gz'))
    save_npy2nii(Body_psdm, [2.0, 2.5, 2.5],
                 os.path.join(save_dir, ID, 'PSDM', 'Body.nii.gz'))

    # isVMAT
    isVMAT_df = pd.DataFrame({'isVMAT': [data_npz['isVMAT']]})
    isVMAT_df.to_csv(os.path.join(save_dir, ID, 'isVMAT.csv'), index=False)
    return full_name

if __name__ == "__main__":
    full_name_list_HaN = os.listdir(r'train_data/train_HaN/train')
    with Pool(8) as pool:
        results = pool.map(npy2nii_processing_HaN, full_name_list_HaN)
    print(results)

    full_name_list_Lung = os.listdir(r'train_data/train_Lung/train')
    with Pool(8) as pool:
        results = pool.map(npy2nii_processing_Lung, full_name_list_Lung)
    print(results)