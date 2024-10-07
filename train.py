import os

import torch

#import CTformer
import ESAU_model
import Moga
import redcnn
import statistics
import yaml
import edcnn_model
import measure
from preloss import WGAN_VGG
import save_crop_image
import save_image
from utils import network_parameters
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import utils
import numpy as np
import random
import preloss
from data_RGB import get_training_data, get_validation_data

from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model.SUNet import SUNet_model

## Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


def denormalize_(image):
    image = image * (0.0 + 255.0) + 0.0
    return image

def normalize(image):
    image = (image + 0.0) / (0.0 + 255.0)
    return image
def denormalize_ct(image):
    image = image * (1024.0 + 3072.0) - 1024.0
    return image

def normalize_ct(image):
    image = (image + 1024.0) / (1024.0 + 3072.0)
    return image


def trunc(mat):
    mat[mat <= -160] = -160
    mat[mat >= 240] = 240
    return mat
def trunc_train(mat):
    mat[mat <= -1024] = -1024.0
    mat[mat >= 3072] = 3072.0
    return  mat
## Load yaml configuration file
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']

gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
## Build Model
log_dir = os.path.join(Train['SAVE_DIR'], 'log')
img_path = os.path.join('image','LP','edcnn')
file_path = os.path.join(log_dir, 'lp.txt')
print('==> Build the model')
#model_restored = SUNet_model(opt)
#model_restored = CTformer.CTformer()
model_restored = edcnn_model.EDCNN()
#model_restored = Moga.Moga()
#model_restored = redcnn.RED_CNN()
#model_restored = ESAU_model.ESAU()
#ploss = WGAN_VGG()
#ploss.cuda()
p_number = network_parameters(model_restored)
model_restored.cuda()
#test=np.load(os.path.join('C:/科研/SUNET/SUNet-main/SUNet-main/L506_0_input.npy')) #add
## Training model path direction
mode = opt['MODEL']['MODE']

model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
utils.mkdir(model_dir)
train_dir = Train['TRAIN_DIR']
val_dir = Train['VAL_DIR']

# GPU
gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
print(gpus)
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
if len(device_ids) > 1:
    model_restored = nn.DataParallel(model_restored, device_ids=device_ids)
    #ploss = nn.DataParallel(ploss,device_ids=device_ids)


## Log
#log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

## Optimizer
start_epoch = 0
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

## Scheduler (Strategy)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

## Resume (Continue training by a pretrained model)
if Train['RESUME']:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restored, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')

## Loss
# L1_loss = nn.L1Loss()
MSE_loss =nn.MSELoss()
#ploss = WGAN_VGG().cuda()  #add

## DataLoaders
print('==> Loading datasets')
train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']})
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                          shuffle=True, num_workers=0, drop_last=False)

val_dataset = get_validation_data(val_dir, {'patch_size': Train['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False)
#device_ids=0 #add
# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
    Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
    Model parameters:   {p_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    GPU:                {'GPU' + str(device_ids)}''')
print('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
best_psnr = 0
best_ssim = 0
best_rmse = 1000
best_epoch_psnr = 0
best_epoch_ssim = 0
best_epoch_rmse = 1000
total_start_time = time.time()

def append_results_to_txt(epoch, psnr, ssim, rmse, elapsed_time,std_psnr,std_ssim,std_rmse,file_path):
    result_str = "epoch %d PSNR:%.5f SSIM:%.5f RMSE:%.5f elapsed_time:%.5f std_psnr:%.5f std_ssim:%.5f std_rmse:%.5f" % (epoch, psnr, ssim, rmse,elapsed_time,std_psnr,std_ssim,std_rmse)
    with open(file_path, 'a') as file:
        file.write(result_str + '\n')

a = 0
b = 1
c = 0
for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0.0
    perceptualloss = 0.0
    ssim_loss = 0.0
    mseloss = 0.0
    train_id = 1

    model_restored.train()
    #data  [-1024......]
    for i, data in enumerate(tqdm(train_loader), 0):
        # Forward propagation
        for param in model_restored.parameters():
            param.grad = None
        target = data[0].cuda()
        input_ = data[1].cuda()
        #input_ = trunc(input_)
        #target = trunc(target)
        input_ = normalize_ct(input_)
        target = normalize_ct(target)
        #input_ = denormalize_(input_)
        #target = denormalize_(target)
        restored = model_restored(input_)
        #save_crop_image.save_fig(input_.view(768, 768), target.view(768, 768), restored.view(768, 768), i, epoch)
        # if i>=0:
        #      break
        #target = trunc_train(target) #add
        #input_ = trunc_train(input_) #add
        # target = normalize(target)
        # input_ = normalize(input_)

        target = target.float()
        input_ = input_.float()
        #restored = model_restored(input_)

        #save_image.save_image(restored,i) #add

        # Compute loss
        #loss = Charbonnier_loss(restored, target)
        #loss = L1_loss(restored, target)
        #print("restored{} target{} ".format(restored.dtype,target.dtype))
        #loss = MSE_loss(restored,target)

        #restored_n = normalize(restored)   #add
        #target_n = normalize(target)        #add
        #perloss = ploss.module.p_loss(restored_n,target_n)
        mse = MSE_loss(restored,target)
        ssimloss = 1 - measure.compute_SSIM(restored,target,3000)
        loss =  b * mse + c * ssimloss  #add
        perloss = 0.0
        #print(loss.dtype)
        perceptualloss += a * perloss   #add
        mseloss += b * mse
        ssim_loss += c * ssimloss

        # Back propagation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    writer.add_scalar('val/perceptualloss', perceptualloss, epoch)  #add
    writer.add_scalar('val/mseloss', mseloss, epoch)    #add
    writer.add_scalar('val/loss', epoch_loss, epoch)    #add
    writer.add_scalar('val/ssimloss',ssim_loss,epoch)


    elapsed_time =0.0
    count = 0
    ## Evaluation (Validation)
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model_restored.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []
        rmse_val=[]
        if Train['val']:
            checkpoint_dir = os.path.join('checkpoints','CT','redcnn','Denoising','models','model_latest.pth')
            checkpoint = torch.load(checkpoint_dir)
            model_state_dict = checkpoint['state_dict']
            model_restored.load_state_dict(model_state_dict)
        for ii, data_val in enumerate(tqdm(val_loader), 0):
            target = data_val[0].cuda()
            target = target.float()
            #target = trunc_train(target)
            input_ = data_val[1].cuda()
            input_ = input_.float()
            #input_ = trunc_train(input_)
            input_ = normalize_ct(input_)
            target = normalize_ct(target)
            with torch.no_grad():
                start_time = time.time()
                restored = model_restored(input_)
                end_time = time.time()
                count += 1
                elapsed_time += end_time - start_time
            input_ = input_.float()  #add
            input_ = denormalize_ct(input_)
            target = denormalize_ct(target)
            restored = denormalize_ct(restored)
            input_ = trunc(input_)
            target = trunc(target)
            restored = trunc(restored)
            # input = trunc(denormalize_(input_.view(512,512)))
            # target_=trunc(denormalize_(target.view(512,512)))
            # restored_ = trunc(denormalize_(restored.view(512,512)))
            if ii%100 == 0:
                save_crop_image.save_fig(input_.view(512,512),target.view(512,512),restored.view(512,512),ii,epoch,img_path)
                #save_crop_image.savefignew(input_.view(512,512),target.view(512,512),restored.view(512,512),ii,epoch)

            for res, tar in zip(input_, target):
                #psnr_val_rgb.append(utils.torchPSNR(res, tar))
                psnr_val_rgb.append(measure.compute_PSNR(restored,target,400.0)) #add
                #ssim_val_rgb.append(utils.torchSSIM(restored, target))
                ssim_val_rgb.append(measure.compute_SSIM(restored,target,400.0)) #add
                rmse_val.append(measure.compute_RMSE(restored,target))
                #ssim_val_rgb.append(utils.torchSSIM(target, target))  #add
        std_psnr = statistics.stdev(psnr_val_rgb)
        std_ssim = statistics.stdev(ssim_val_rgb)
        std_rmse = statistics.stdev(rmse_val)

        elapsed_time = elapsed_time / count
        psnr_val = np.mean(psnr_val_rgb)
        ssim_val = np.mean(ssim_val_rgb)
        rmse_val = np.mean(rmse_val)
        append_results_to_txt(epoch,psnr_val,ssim_val,rmse_val,elapsed_time,std_psnr,std_ssim,std_rmse,file_path)
        print("epoch %d PSNR:%.5f SSIM:%.5f RMSE:%.5f"%(epoch,psnr_val,ssim_val,rmse_val))

        # Save the best PSNR model of validation
        if psnr_val > best_psnr:
            best_psnr = psnr_val
            best_epoch_psnr = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestPSNR.pth"))
        print("[epoch %d PSNR: %.5f --- best_epoch %d Best_PSNR %.5f]" % (epoch, psnr_val, best_epoch_psnr, best_psnr))

        # Save the best SSIM model of validation
        if ssim_val > best_ssim:
            best_ssim = ssim_val
            best_epoch_ssim = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestSSIM.pth"))
        print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
            epoch, ssim_val, best_epoch_ssim, best_ssim))

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_epoch_rmse = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestRMSE.pth"))
        print("[epoch %d RMSE: %.3f --- best_epoch %d Best_RMSE %.3f]" % (
            epoch, rmse_val, best_epoch_rmse, best_rmse))

        """ 
        # Save evey epochs of model
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
        """

        writer.add_scalar('val/PSNR', psnr_val, epoch)
        writer.add_scalar('val/SSIM', ssim_val, epoch)
        writer.add_scalar('val/rmse',rmse_val,epoch)
    scheduler.step()

    print("------------------------------------------------------------------")
    print('a: {}   b:{}  c:{}'.format(a,b,c))
    print("Epoch: {}\tTime: {:.4f}\tLoss: {}\t perceptualloss: {}\t mseloss:{}\t ssimloss:{}\t LearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss,perceptualloss,mseloss,ssim_loss,scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    #writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
writer.close()

total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
