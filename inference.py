import os
import numpy as np
import torch
import argparse
import shutil
from guided_diffusion.unet import UNetModel
import math
from skimage.morphology import binary_dilation, ball, dilation
from utils import _get_gaussian, clip_norm, PSDM_transform, get_combine_oar_mask, get_combine_ptv_mask
import json
from scipy import ndimage
import time
import pandas as pd
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='which gpu is used')
parser.add_argument('--net', type=int, default='1', help='net')
parser.add_argument('--model_path', type=str, default='trained_models/net1_size_96_96_128_input1_aug1_norm1_loss1_optim1_seed42/bs3_epoch500_fold5/epoch500.pth', help='trained model path')
parser.add_argument('--bs', type=int, default=36, help='batchsize')
parser.add_argument('--tta', type=int, default=1, help='TTA')
parser.add_argument('--input_size', type=int, nargs='*', default=(96, 96, 128), help='input_size')
parser.add_argument('--input_mode', type=int, default=1, help='input_mode')
parser.add_argument('--norm_mode', type=int, default=1, help='norm_mode')
parser.add_argument('--stride_rate', type=float, default=0.5, help='stride_rate')
parser.add_argument('--gauss', type=int, default=1, help='gauss')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

df = pd.read_csv(glob.glob(r'data/meta_data*.csv')[0])
df['npz_path'] = df['npz_path'].apply(os.path.basename)
data_path_list = glob.glob(r'data/*.npz')

scale_dose_Dict = json.load(open(r'data/PTV_DICT.json', 'r'))
pat_obj_dict = json.load(open(r'data/Pat_Obj_DICT.json', 'r'))

new_dir = 'results'
if os.path.exists(new_dir):
    shutil.rmtree(new_dir)
os.makedirs(new_dir, exist_ok=True)

HaN_OAR_LIST = ['Cochlea_L', 'Cochlea_R', 'Eyes', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R', 'Chiasim',
                 'LacrimalGlands', 'BrachialPlexus', 'Brain',  'BrainStem_03',  'Esophagus', 'Lips', 'Lungs', 'Trachea',
                 'Posterior_Neck', 'Shoulders', 'Larynx-PTV', 'Mandible-PTV', 'OCavity-PTV', 'ParotidCon-PTV',
                 'Parotidlps-PTV', 'Parotids-PTV', 'PharConst-PTV', 'Submand-PTV', 'SubmandL-PTV', 'SubmandR-PTV',
                 'Thyroid-PTV', 'SpinalCord_05']
HaN_OAR_DICT = {HaN_OAR_LIST[i]: (i+1) for i in range(len(HaN_OAR_LIST))}
Lung_OAR_LIST = ["PTV_Ring.3-2", "Total Lung-GTV", "SpinalCord",  "Heart",  "LAD", "Esophagus",  "BrachialPlexus",
                 "GreatVessels", "Trachea", "Body_Ring0-3"]
Lung_OAR_DICT = {Lung_OAR_LIST[i]: (i+1) for i in range(len(Lung_OAR_LIST))}

patch_size = args.input_size
stride = (int(args.input_size[0] * args.stride_rate),
          int(args.input_size[1] * args.stride_rate),
          int(args.input_size[2] * args.stride_rate))

if args.norm_mode == 1:
    CT_clip = (-1000, 1500)
    dose_clip = (0, 75)
    norm_range = (-1, 1)
    oar_div_factor = 30.
    beam_div_factor = 90.

if args.tta:
    T = 3
else:
    T = 1

if args.input_mode in [1]:
    in_channels = 9

if args.net == 1:
    net = UNetModel(image_size=args.input_size, in_channels=in_channels, model_channels=32, out_channels=1, num_res_blocks=2,
                    attention_resolutions=(32,), channel_mult=(0.5, 1, 2, 4, 8), dims=3, num_heads=4, use_fp16=True).half().cuda()
elif args.net == 5:
    net = UNetModel(image_size=args.input_size, in_channels=in_channels, model_channels=32, out_channels=1,
                    num_res_blocks=2,
                    attention_resolutions=(32,), channel_mult=(1, 1, 2, 4, 8), dims=3, num_heads=4,
                    use_fp16=True).half().cuda()

gaussian_importance_map = _get_gaussian(patch_size, sigma_scale=1. / 8)

net.load_state_dict(torch.load(args.model_path))
net.eval()

structuring_element = ball(3)
cylinder_kernel = np.zeros((3, 11, 11), dtype=np.uint8)
cylinder_kernel[1] = ball(5)[5]

with torch.no_grad():
    for i, full_path in enumerate(data_path_list):
        full_name = os.path.basename(full_path)
        site = int(df.loc[df.npz_path == full_name, 'site'].values[0])
        ID = full_name.replace('.npz', '')
        print(i, ID)
        PatientID = ID.split('+')[0]
        if len(str(PatientID)) < 3:
            PatientID = f"{PatientID:0>3}"
        data_npz = np.load(full_path, allow_pickle=True)
        data_npz = dict(data_npz)['arr_0'].item()

        CT = data_npz['img']
        img_shape = CT.shape
        input_img = [clip_norm(CT, CT_clip, norm_range)[np.newaxis, ...]]

        if args.input_mode == 1:
            # OAR mask
            if site == 1:
                OAR_DICT = HaN_OAR_DICT
            elif site == 2:
                OAR_DICT = Lung_OAR_DICT
            try:
                need_list = pat_obj_dict[ID.split('+')[0]]
            except:
                if site == 1:
                    need_list = HaN_OAR_LIST
                elif site == 2:
                    need_list = Lung_OAR_LIST
            input_one = get_combine_oar_mask(data_npz, need_list, OAR_DICT)
            input_img.append(input_one[np.newaxis, ...] / oar_div_factor)
            # comb_optptv mask
            opt_dose_dict = {}
            dose_dict = {}
            for key in scale_dose_Dict[PatientID].keys():
                if key in ['PTV_High', 'PTV_Mid', 'PTV_Low']:
                    opt_dose_dict[scale_dose_Dict[PatientID][key]['OPTName']] = scale_dose_Dict[PatientID][key][
                        'PDose']
                    if key != 'PTV_High':
                        dose_dict[scale_dose_Dict[PatientID][key]['StructName']] = scale_dose_Dict[PatientID][key][
                            'PDose']
                    else:
                        dose_dict[scale_dose_Dict[PatientID][key]['OPTName']] = scale_dose_Dict[PatientID][key][
                            'PDose']
            input_one = get_combine_ptv_mask(data_npz, opt_dose_dict)
            input_img.append(clip_norm(input_one, dose_clip, norm_range)[np.newaxis, ...])
            # comb_ptv mask
            input_one = get_combine_ptv_mask(data_npz, dose_dict)
            input_img.append(clip_norm(input_one, dose_clip, norm_range)[np.newaxis, ...])
            # isocenter PSDM
            isocenter = data_npz['isocenter']
            input_one = np.zeros(data_npz['img'].shape)
            input_one[int(isocenter[0]), int(isocenter[1]), int(isocenter[2])] = 1
            input_one = binary_dilation(input_one, structuring_element)
            input_one = PSDM_transform(mask=input_one, spacing=[2.0, 2.5, 2.5])
            input_img.append(input_one[np.newaxis, ...])
            # angle_plate mask
            input_one = np.zeros(img_shape)
            z_begin = int(isocenter[0]) - 5
            z_end = int(isocenter[0]) + 5
            z_begin = max(0, z_begin)
            z_end = min(input_one.shape[0], z_end)
            D3_plate = np.repeat(data_npz['angle_plate'][np.newaxis, :, :], z_end - z_begin, axis=0)
            if D3_plate.shape[1] != input_one.shape[1] or D3_plate.shape[2] != input_one.shape[2]:
                D3_plate = ndimage.zoom(D3_plate, (
                    1, input_one.shape[1] / D3_plate.shape[1],
                    input_one.shape[2] / D3_plate.shape[2]), order=0)
            input_one[z_begin: z_end] = D3_plate
            input_img.append(input_one[np.newaxis, ...])
            # angle_plate PSDM
            input_one = PSDM_transform(mask=input_one, spacing=[2.0, 2.5, 2.5])
            input_img.append(input_one[np.newaxis, ...])
            # beam_plate mask
            input_one = data_npz['beam_plate']
            input_img.append(input_one[np.newaxis, ...] / beam_div_factor)
            # Body mask
            Body = data_npz['Body']
            input_img.append(Body[np.newaxis, ...])
        input_img = np.concatenate(input_img, axis=0)
        Body_dilation = dilation(Body, cylinder_kernel)

        isVMAT = data_npz['isVMAT']
        if (not isVMAT) and (site == 1):
            isVMAT = torch.tensor([0])
        elif isVMAT and (site == 1):
            isVMAT = torch.tensor([1])
        elif (not isVMAT) and (site == 2):
            isVMAT = torch.tensor([2])
        elif isVMAT and (site == 2):
            isVMAT = torch.tensor([3])

        original_shape = input_img.shape[1:]
        pad_z1 = 0
        pad_x1 = 0
        pad_y1 = 0
        if original_shape[0] < patch_size[0]:
            pad_z = patch_size[0] - original_shape[0]
            pad_z1 = pad_z // 2
            pad_z2 = pad_z - pad_z1
            input_img = np.pad(input_img, ((0, 0), (pad_z1, pad_z2), (0, 0), (0, 0)), 'constant')
        if original_shape[1] < patch_size[1]:
            pad_x = patch_size[1] - original_shape[1]
            pad_x1 = pad_x // 2
            pad_x2 = pad_x - pad_x1
            input_img = np.pad(input_img, ((0, 0), (0, 0), (pad_x1, pad_x2), (0, 0)), 'constant')
        if original_shape[2] < patch_size[2]:
            pad_y = patch_size[2] - original_shape[2]
            pad_y1 = pad_y // 2
            pad_y2 = pad_y - pad_y1
            input_img = np.pad(input_img, ((0, 0), (0, 0), (0, 0), (pad_y1, pad_y2)), 'constant')
        new_shape = input_img.shape[1:]

        prediction = np.zeros(new_shape)
        repeat = np.zeros(new_shape)

        num_z = 1 + math.ceil((new_shape[0] - patch_size[0]) / stride[0])
        num_x = 1 + math.ceil((new_shape[1] - patch_size[1]) / stride[1])
        num_y = 1 + math.ceil((new_shape[2] - patch_size[2]) / stride[2])

        total_num = num_z * num_x * num_y
        indexs_all = []
        img_patches_all = []
        isVMAT_all = []
        t_all = []
        for z in range(num_z):
            for x in range(num_x):
                for y in range(num_y):
                    n_patch = z * num_x * num_y + x * num_y + y + 1
                    x_left = x * stride[1]
                    x_right = x * stride[1] + patch_size[1]
                    y_up = y * stride[2]
                    y_down = y * stride[2] + patch_size[2]
                    z_top = z * stride[0]
                    z_botton = z * stride[0] + patch_size[0]
                    if x == num_x - 1:
                        x_left = new_shape[1] - patch_size[1]
                        x_right = new_shape[1]
                    if y == num_y - 1:
                        y_up = new_shape[2] - patch_size[2]
                        y_down = new_shape[2]
                    if z == num_z - 1:
                        z_top = new_shape[0] - patch_size[0]
                        z_botton = new_shape[0]
                    img_one_original = input_img[:, z_top:z_botton, x_left:x_right, y_up:y_down]
                    for t in range(T):
                        if t == 0:
                            img_one_tta = img_one_original
                        elif t == 1:
                            img_one_tta = np.flip(img_one_original, axis=-1)
                        elif t == 2:
                            img_one_tta = np.flip(img_one_original, axis=-2)
                        img_one_tta = np.ascontiguousarray(img_one_tta)
                        indexs_all.append([z_top, z_botton, x_left, x_right, y_up, y_down])
                        img_patches_all.append(np.copy(img_one_tta[np.newaxis, ...]))
                        isVMAT_all.append(isVMAT)
                        t_all.append(t)
                        num_path = (n_patch - 1) * T + (t + 1)
                        if ((num_path % args.bs) == 0) or (num_path == total_num * T):
                            img_one = np.concatenate(img_patches_all, axis=0)
                            img_one = torch.from_numpy(img_one).half().cuda()
                            isVMAT_all = torch.cat(isVMAT_all, dim=0).half().cuda()

                            # start = time.time()
                            # for _ in range(20):
                            #     pred_one_batch = net(img_one, isVMAT_all)
                            # end = time.time()
                            # print(f"Time taken for average forward pass: {(end - start) / 20:.4f} seconds")

                            pred_one_batch = net(img_one, isVMAT_all)

                            pred_one_batch = torch.tanh(pred_one_batch).cpu().detach().numpy()
                            for b, index in enumerate(indexs_all):
                                z_top, z_botton, x_left, x_right, y_up, y_down = index
                                pred_one = pred_one_batch[b, 0]
                                t_ = t_all[b]
                                if t_ == 1:
                                    pred_one = np.flip(pred_one, axis=-1)
                                elif t_ == 2:
                                    pred_one = np.flip(pred_one, axis=-2)
                                if args.gauss:
                                    prediction[z_top:z_botton, x_left:x_right,
                                    y_up:y_down] += pred_one * gaussian_importance_map
                                    repeat[z_top:z_botton, x_left:x_right, y_up:y_down] += gaussian_importance_map
                                else:
                                    repeat_one = np.ones(pred_one.shape)
                                    prediction[z_top:z_botton, x_left:x_right, y_up:y_down] += pred_one
                                    repeat[z_top:z_botton, x_left:x_right, y_up:y_down] += repeat_one
                            indexs_all = []
                            img_patches_all = []
                            isVMAT_all = []
                            t_all = []
        repeat = repeat.astype(np.float32)
        prediction = prediction / repeat
        prediction = (prediction + 1) / 2 * (dose_clip[1] - dose_clip[0]) + dose_clip[0]
        prediction = prediction[pad_z1:pad_z1+original_shape[0],
                     pad_x1:pad_x1+original_shape[1],
                     pad_y1:pad_y1+original_shape[2]]

        prediction *= Body_dilation
        np.save(os.path.join(new_dir, '{}_pred.npy'.format(ID)), prediction)