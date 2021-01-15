import os
import numpy as np
import gc
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from loss_function import *
from network import TinyDCU_Net_16
import csv
from istft import ISTFT
from utils.signalprocess import analysis_window


def wSDRLoss(mixed, clean, clean_est, eps=2e-7):
    # Used on signal level(time-domain). Backprop-able istft should be used.
    # Batched audio inputs shape (N x T) required.
    bsum = lambda x: torch.sum(x, dim=1)  # Batch preserving sum for convenience.

    def mSDRLoss(orig, est):
        # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
        # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
        #  > Maximize Correlation while producing minimum energy output.
        correlation = bsum(orig * est)
        energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
        return -(correlation / (energies + eps))

    noise = mixed - clean
    noise_est = mixed - clean_est
    a = bsum(clean ** 2) / (bsum(clean ** 2) + bsum(noise ** 2) + eps)
    wSDR = a * mSDRLoss(clean, clean_est) + (1 - a) * mSDRLoss(noise, noise_est)
    return torch.mean(wSDR)


class Solver(object):
    def __init__(self, config, train_reader, valid_reader, test_reader):
        # Data reader
        self.train_reader = train_reader
        self.valid_reader = valid_reader
        self.test_reader = test_reader
        self.narray_window = analysis_window(config.frame_size, config.frame_shift)
        self.tensor_window = torch.tensor(self.narray_window, dtype=torch.float32).to(self.device)
        self.stft = lambda x: torch.stft(x, config.frame_size, config.frame_shift, window=self.tensor_window)
        self.istft = ISTFT(config.frame_size, config.frame_shift, window=self.narray_window).to(self.device)
        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.MSELoss()
        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        # User-parameters
        self.half_lr = config.half_lr
        self.prev_cv_loss = float("inf")
        self.best_cv_loss = float("inf")
        self.best_tr_loss = float("inf")
        self.having = False
        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size
        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step
        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        # Model
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type == 'TinyDCU_Net_16':
            self.unet = TinyDCU_Net_16()

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2])
        self.unet.to(self.device)
        self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def train(self):
        """Train encoder, generator and discriminator."""
        # ====================================== Training ===========================================#
        # ===========================================================================================#
        # Net Train
        lr = self.lr
        start_epoch = -1
        if not os.path.isdir("./models/checkpoint"):
            os.mkdir("./models/checkpoint")
        checkpoint_path = './models/checkpoint/%s_ckpt_best.pth' % self.model_type  # 断点路径
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)  # 加载断点
            self.unet.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
            self.optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            start_epoch = checkpoint['epoch']

        for epoch in range(start_epoch + 1, self.num_epochs):
            self.unet.train(True)
            train_loss = 0
            start = time.time()
            while self.train_reader.is_running_out() is not True:
                # network forword processing
                if self.model_type == 'TinyDCU_Net_16':
                    noisy_sig, clean_sig, frame_list = self.train_reader.load_one_batch()
                    noisy_sig = noisy_sig.to(self.device)
                    clean_sig = clean_sig.to(self.device)
                    noisy_stft = self.stft(noisy_sig)
                    noisy_stft = noisy_stft.permute(0, 3, 2, 1).contiguous()
                    # network forword processing
                    estimate_real, estimate_imag = self.unet(noisy_stft[:, 0, :, :].unsqueeze(1),
                                                             noisy_stft[:, 1, :, :].unsqueeze(1))
                    estimate_real = estimate_real.permute(0, 3, 2, 1).contiguous()
                    estimate_imag = estimate_imag.permute(0, 3, 2, 1).contiguous()
                    estimate_real, estimate_imag = torch.squeeze(estimate_real, 3), torch.squeeze(estimate_imag, 3)
                    estimate_sig = self.istft(estimate_real, estimate_imag, noisy_sig.size(1))
                    estimate_sig = torch.squeeze(estimate_sig, dim=1)
                    loss = wSDRLoss(noisy_sig, clean_sig, estimate_sig)
                    train_loss += loss.item()
                    # Backprop + optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # Print the log info
            self.train_reader.reset()
            print("[Trainning] [%d/%d], Elipse Time: %4f, Train Loss: %4f" % (
                epoch + 1, self.num_epochs, time.time() - start, train_loss))
            if train_loss < self.best_tr_loss:
                self.best_tr_loss = train_loss
                best_epoch = epoch
                best_unet_path = os.path.join(self.model_path, '%s-%d-%d-train.pkl' % (
                    self.model_type, self.num_epochs, best_epoch))
                best_unet = self.unet.state_dict()
                print('Find better train model, Best %s model loss : %.4f' % (self.model_type, train_loss))
                torch.save(best_unet, best_unet_path)
                checkpoint = {
                    "net": self.unet.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    "epoch": epoch
                }
                torch.save(checkpoint, checkpoint_path)

            # ===================================== Cross Validation =================================#
            start = time.time()
            self.unet.train(False)
            self.unet.eval()
            valid_loss = 0
            while self.valid_reader.is_running_out() is not True:
                noisy_feat, clean_feat, noisy_sig, clean_sig = self.valid_reader.load_one_item()
                noisy_feat = torch.tensor(noisy_feat, dtype=torch.float32).to(self.device)
                clean_feat = torch.tensor(clean_feat, dtype=torch.float32).to(self.device)
                # network forword processing
                if self.model_type == 'TinyDCU_Net_16':
                    clean_flat = clean_feat.view(clean_feat.size(0), -1)
                    estimate_real, estimate_imag = self.unet(noisy_feat[:, 0, :, :].unsqueeze(1),
                                                             noisy_feat[:, 1, :, :].unsqueeze(1))
                    estimate_feat = torch.cat([estimate_real, estimate_imag], 1)
                    estimate_flat = estimate_feat.view(estimate_feat.size(0), -1)
                    loss = self.criterion(estimate_flat, clean_flat)
                    valid_loss += loss.item()

            # Print the log info
            self.valid_reader.reset()
            print('[Validation] [%d/%d], Elipse Time: %4f, Validation Loss: %.4f' % (
                epoch + 1, self.num_epochs, time.time() - start, valid_loss))
            # Decay learning rate
            if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print('Decay learning rate to lr: {}.'.format(lr))

            if valid_loss < self.best_cv_loss:
                self.best_cv_loss = valid_loss
                best_epoch = epoch
                best_unet_path = os.path.join(self.model_path, '%s-%d-%d-valid.pkl' % (
                    self.model_type, self.num_epochs, best_epoch))
                best_unet = self.unet.state_dict()
                print('Find better valid model, Best %s model loss : %.4f\n' % (self.model_type, valid_loss))
                torch.save(best_unet, best_unet_path)
