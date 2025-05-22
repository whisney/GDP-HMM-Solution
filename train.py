from guided_diffusion.unet import UNetModel
import os
from dataset import Dataset_Dose
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch.nn as nn
import numpy as np
import torch
import math
import shutil
from torchvision.utils import make_grid
import argparse
from torch.optim.lr_scheduler import MultiStepLR
import random
from loss_function.pytorch_loss_function import VGGLoss_3D
from utils import poly_lr_scheduler

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
parser.add_argument('--bs', type=int, default=3, help='batch size')
parser.add_argument('--epoch', type=int, default=500, help='all_epochs')
parser.add_argument('--net', type=int, default=1, help='net')
parser.add_argument('--fold', type=int, default=5, help='0/1/2/3/4')
parser.add_argument('--input_size', type=int, nargs='*', default=(96, 96, 128), help='input_size')
parser.add_argument('--input_mode', type=int, default=1, help='input_mode')
parser.add_argument('--aug_mode', type=int, default=1, help='aug_mode')
parser.add_argument('--norm_mode', type=int, default=1, help='norm_mode')
parser.add_argument('--loss_mode', type=int, default=1, help='loss_mode')
parser.add_argument('--optim_mode', type=int, default=1, help='optim_mode')
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_dir = r'NII_data'
split_path = r'NII_data/all_split_5fold_seed0.pkl'

save_dir = r'trained_models/net{}_size_{}_{}_{}_input{}_aug{}_norm{}_loss{}_optim{}_seed{}/bs{}_epoch{}_fold{}'.format(
    args.net, args.input_size[0], args.input_size[1], args.input_size[2], args.input_mode, args.aug_mode,
    args.norm_mode, args.loss_mode, args.optim_mode, args.seed, args.bs, args.epoch, args.fold)
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)
train_writer = SummaryWriter(os.path.join(save_dir, 'log/train'), flush_secs=2)
val_writer = SummaryWriter(os.path.join(save_dir, 'log/val'), flush_secs=2)
print(save_dir)

train_data = Dataset_Dose(data_dir=data_dir, split_file=split_path, fold=args.fold, subset='train', aug=True,
                          norm_mode=args.norm_mode, input_mode=args.input_mode, aug_mode=args.aug_mode, size=args.input_size)
train_dataloader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)
val_data = Dataset_Dose(data_dir=data_dir, split_file=split_path, fold=args.fold, subset='val', aug=False,
                          norm_mode=args.norm_mode, input_mode=args.input_mode, aug_mode=args.aug_mode, size=args.input_size)
val_dataloader = DataLoader(dataset=val_data, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)

if args.norm_mode == 1:
    dose_clip = (0, 75)

if args.input_mode in [1]:
    in_channels = 9

if args.net == 1:
    net = UNetModel(image_size=args.input_size, in_channels=in_channels, model_channels=32, out_channels=1, num_res_blocks=2,
                    attention_resolutions=(32,), channel_mult=(0.5, 1, 2, 4, 8), dims=3, num_heads=4, use_fp16=False).cuda()
elif args.net == 2:
    net = UNetModel(image_size=args.input_size, in_channels=in_channels, model_channels=32, out_channels=1, num_res_blocks=2,
                    attention_resolutions=(32, 16), channel_mult=(0.5, 1.5, 3, 6, 10), dims=3, num_heads=4, use_fp16=False).cuda()
elif args.net == 3:
    net = UNetModel(image_size=args.input_size, in_channels=in_channels, model_channels=32, out_channels=1, num_res_blocks=2,
                    attention_resolutions=(), channel_mult=(0.5, 2, 4, 6, 10), dims=3, num_heads=4, use_fp16=False).cuda()
elif args.net == 4:
    net = UNetModel(image_size=args.input_size, in_channels=in_channels, model_channels=32, out_channels=1, num_res_blocks=2,
                    attention_resolutions=(32, 16, 8), channel_mult=(1, 2, 4, 6, 10), dims=3, num_heads=4, use_fp16=False).cuda()
elif args.net == 5:
    net = UNetModel(image_size=args.input_size, in_channels=in_channels, model_channels=32, out_channels=1, num_res_blocks=2,
                    attention_resolutions=(32,), channel_mult=(1, 1, 2, 4, 8), dims=3, num_heads=4, use_fp16=False).cuda()

train_data_len = train_data.len
val_data_len = val_data.len
print('train_lenth: %i  val_lenth: %i' % (train_data_len, val_data_len))

MAEloss = nn.L1Loss(reduction='none')
VGGloss = VGGLoss_3D()

if args.loss_mode == 1:
    masked_MAEloss = True
    MAEloss_weight = 20
    VGGloss_weight = 1
elif args.loss_mode == 2:
    masked_MAEloss = True
    MAEloss_weight = 1
    VGGloss_weight = 0
elif args.loss_mode == 3:
    masked_MAEloss = True
    MAEloss_weight = 20
    VGGloss_weight = 0.5

if args.optim_mode == 1:
    lr_max = 0.0002
    optimizer = optim.AdamW(net.parameters(), lr=lr_max, weight_decay=0.0001)
    lr_strategy = 'step'
elif args.optim_mode == 2:
    lr_max = 0.01
    optimizer = optim.SGD(net.parameters(), lr=lr_max, weight_decay=0.0001, momentum=0.99, nesterov=True)
    lr_strategy = 'step'

scheduler_step = MultiStepLR(optimizer, milestones=[int((7 / 10) * args.epoch),
                                                   int((9 / 10) * args.epoch)], gamma=0.1, last_epoch=-1)
best_MAE = 1000
best_SSIM = 0
print('training')

for epoch in range(args.epoch):
    if lr_strategy == 'ploy':
        poly_lr_scheduler(optimizer, lr_max, epoch, lr_decay_iter=1, max_iter=args.epoch, power=0.9)
    lr = optimizer.param_groups[0]['lr']
    epoch_train_total_loss = []
    epoch_train_MAE = []
    for i, (input_img, dose, isVMAT, mask_for_loss) in enumerate(train_dataloader):
        if mask_for_loss.sum() == 0:
            continue
        input_img, dose, isVMAT, mask_for_loss = input_img.float().cuda(), dose.float().cuda(), \
                                                 isVMAT.float().cuda(), mask_for_loss.float().cuda()

        net.train()
        optimizer.zero_grad()
        pred_dose = torch.tanh(net(input_img, isVMAT))

        if masked_MAEloss:
            MAEloss_final = torch.sum(MAEloss(pred_dose, dose) * mask_for_loss) / mask_for_loss.sum()
        else:
            MAEloss_final = torch.mean(MAEloss(pred_dose, dose))
        if VGGloss_weight > 0:
            VGGloss_final = VGGloss(pred_dose.repeat(1, 3, 1, 1, 1), dose.repeat(1, 3, 1, 1, 1))
            total_loss = MAEloss_weight * MAEloss_final + VGGloss_weight * VGGloss_final
        else:
            total_loss = MAEloss_weight * MAEloss_final
        total_loss.backward()
        optimizer.step()

        pred_dose = pred_dose.detach().cpu()
        Dose_predictions = (pred_dose.numpy() + 1) / 2 * (dose_clip[1] - dose_clip[0]) + dose_clip[0]
        Dose_real = (dose.cpu().numpy() + 1) / 2 * (dose_clip[1] - dose_clip[0]) + dose_clip[0]
        Body_mask = input_img[:, -1:].cpu().numpy()
        isodose_5Gy_mask = ((Dose_real > 5) | (Dose_predictions > 5)) & (Body_mask > 0)
        MAE = (np.abs(Dose_predictions - Dose_real) * isodose_5Gy_mask).sum() / isodose_5Gy_mask.sum()
        epoch_train_total_loss.append(total_loss.item())
        epoch_train_MAE.append(MAE)
        print('[%d/%d, %5d/%d] train_total_loss: %.3f MAE: %.3f' %
              (epoch + 1, args.epoch, i + 1, math.ceil(train_data_len / args.bs), total_loss.item(), MAE))

    if lr_strategy == 'step':
        scheduler_step.step()

    net.eval()
    epoch_val_MAE = []
    image_CT = []
    image_dose = []
    pred_dose_show = []
    with torch.no_grad():
        for i, (input_img, dose, isVMAT, mask_for_loss) in enumerate(val_dataloader):
            if mask_for_loss.sum() == 0:
                continue
            input_img, dose, isVMAT, mask_for_loss = input_img.float().cuda(), dose.float().cuda(), \
                                                     isVMAT.float().cuda(), mask_for_loss.float().cuda()

            pred_dose = torch.tanh(net(input_img, isVMAT))
            pred_dose = pred_dose.detach().cpu()
            Dose_predictions = (pred_dose.numpy() + 1) / 2 * (dose_clip[1] - dose_clip[0]) + dose_clip[0]
            Dose_real = (dose.cpu().numpy() + 1) / 2 * (dose_clip[1] - dose_clip[0]) + dose_clip[0]
            Body_mask = input_img[:, -1:].cpu().numpy()
            isodose_5Gy_mask = ((Dose_real > 5) | (Dose_predictions > 5)) & (Body_mask > 0)
            MAE = (np.abs(Dose_predictions - Dose_real) * isodose_5Gy_mask).sum() / isodose_5Gy_mask.sum()
            epoch_val_MAE.append(MAE)

            if i in [2, 4, 6, 8] and epoch % (args.epoch // 20) == 0:
                image_CT.append(input_img[0:1, 0:1, args.input_size[0]//2, :, :].cpu())
                image_dose.append(dose[0:1, :, args.input_size[0]//2, :, :].cpu())
                pred_dose_show.append(pred_dose[0:1, :, args.input_size[0]//2, :, :].cpu())
    epoch_train_total_loss = np.mean(epoch_train_total_loss)
    epoch_train_MAE = np.mean(epoch_train_MAE)
    epoch_val_MAE = np.mean(epoch_val_MAE)
    print(
        '[%d/%d] train_total_loss: %.3f train_MAE: %.3f val_MAE: %.3f'
        % (epoch + 1, args.epoch, epoch_train_total_loss, epoch_train_MAE, epoch_val_MAE))

    if epoch_val_MAE < best_MAE:
        best_MAE = epoch_val_MAE
        torch.save(net.state_dict(), os.path.join(save_dir, 'best_MAE.pth'))
    train_writer.add_scalar('lr', lr, epoch)
    train_writer.add_scalar('total_loss', epoch_train_total_loss, epoch)
    train_writer.add_scalar('MAE', epoch_train_MAE, epoch)

    val_writer.add_scalar('MAE', epoch_val_MAE, epoch)
    val_writer.add_scalar('best_MAE', best_MAE, epoch)
    if epoch % (args.epoch // 20) == 0:
        image_CT = torch.cat(image_CT, dim=0)
        image_dose = torch.cat(image_dose, dim=0)
        pred_dose_show = torch.cat(pred_dose_show, dim=0)
        image_CT = make_grid(image_CT, 2, normalize=True)
        image_dose = make_grid(image_dose, 2, normalize=True)
        pred_dose_show = make_grid(pred_dose_show, 2, normalize=True)
        val_writer.add_image('image_CT', image_CT, epoch)
        val_writer.add_image('image_dose', image_dose, epoch)
        val_writer.add_image('pred_dose_show', pred_dose_show, epoch)
    if (epoch + 1) == args.epoch:
        torch.save(net.state_dict(), os.path.join(save_dir, 'epoch' + str(epoch + 1) + '.pth'))

train_writer.close()
val_writer.close()
print('saved_model_name:', save_dir)