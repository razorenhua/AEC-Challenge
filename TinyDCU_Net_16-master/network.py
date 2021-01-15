import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import math
import sys


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ConvUpBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvUpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ConvPoolBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvPoolBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class CrossResidualBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(CrossResidualBlock, self).__init__()

        self.conv1 = single_conv(ch_in=ch_in, ch_out=ch_out)

    def forward(self, x1, x2, factor):
        x = self.conv1(x2)
        x = F.interpolate(x, size=[int(x2.size()[-2] / factor), int(x2.size()[-1] / factor)], mode='bilinear',
                          align_corners=True)
        x = torch.add(x1, x)
        y = self.conv1(x1)
        y = F.interpolate(y, size=[int(x1.size()[-2] * factor), int(x1.size()[-1] * factor)], mode='bilinear',
                          align_corners=True)
        y = torch.add(x2, y)
        return x, y


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class up_conv_elu(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_elu, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class single_conv_elu(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv_elu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_Chomp_F(nn.Module):
    def __init__(self, chomp_f):
        super(up_Chomp_F, self).__init__()
        self.chomp_f = chomp_f

    def forward(self, x):
        return x[:, :, :, self.chomp_f:]


class down_Chomp_F(nn.Module):
    def __init__(self, chomp_f):
        super(down_Chomp_F, self).__init__()
        self.chomp_f = chomp_f

    def forward(self, x):
        return x[:, :, :, :-self.chomp_f]


class Chomp_T(nn.Module):
    def __init__(self, chomp_t):
        super(Chomp_T, self).__init__()
        self.chomp_t = chomp_t

    def forward(self, x):
        return x[:, :, :-self.chomp_t, :]


class conv_block_elu(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_elu, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 2, 0), value=0.)
        self.pad2 = nn.ConstantPad2d((2, 2, 1, 0), value=0.)
        self.conv = nn.Sequential(
            self.pad1,
            nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 1), bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ELU(inplace=True),
            self.pad2,
            nn.Conv2d(ch_out, ch_out, kernel_size=(2, 5), stride=(1, 2), bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_block_elu(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_block_elu, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.pad2 = nn.ConstantPad2d((2, 2, 1, 0), value=0.)
        self.pad3 = nn.ConstantPad2d((0, 0, 1, 0), value=0.)
        self.up_f = up_Chomp_F(1)
        self.down_f = down_Chomp_F(2)
        self.chomp_t = Chomp_T(1)
        self.conv = nn.Sequential(
            self.pad1,
            nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 1), bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ELU(inplace=True),
            self.pad3,
            nn.ConvTranspose2d(ch_out, ch_out, kernel_size=(2, 5), stride=(1, 2), bias=True),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(ch_out),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# Utility functions for initialization
def complex_rayleigh_init(Wr, Wi, fanin=None, gain=1):
    if not fanin:
        fanin = 1
        for p in Wr.shape[1:]: fanin *= p
    scale = float(gain) / float(fanin)
    theta = torch.empty_like(Wr).uniform_(-math.pi / 2, +math.pi / 2)
    rho = torch.random.rayleigh(scale, tuple(Wr.shape))
    rho = torch.tensor(rho).to(Wr)
    Wr.data.copy_(rho * theta.cos())
    Wi.data.copy_(rho * theta.sin())


# Layers
class ComplexConvWrapper(nn.Module):
    def __init__(self, conv_module, *args, **kwargs):
        super(ComplexConvWrapper, self).__init__()
        self.conv_re = conv_module(*args, **kwargs)
        self.conv_im = conv_module(*args, **kwargs)

    def reset_parameters(self):
        fanin = self.conv_re.in_channels // self.conv_re.groups
        for s in self.conv_re.kernel_size: fanin *= s
        complex_rayleigh_init(self.conv_re.weight, self.conv_im.weight, fanin)
        if self.conv_re.bias is not None:
            self.conv_re.bias.data.zero_()
            self.conv_im.bias.data.zero_()

    def forward(self, xr, xi):
        real = self.conv_re(xr) - self.conv_im(xi)
        imag = self.conv_re(xi) + self.conv_im(xr)
        return real, imag


class ComplexBatchNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(num_features))
            self.Br = torch.nn.Parameter(torch.Tensor(num_features))
            self.Bi = torch.nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br', None)
            self.register_parameter('Bi', None)
        if self.track_running_stats:
            self.register_buffer('RMr', torch.zeros(num_features))
            self.register_buffer('RMi', torch.zeros(num_features))
            self.register_buffer('RVrr', torch.ones(num_features))
            self.register_buffer('RVri', torch.zeros(num_features))
            self.register_buffer('RVii', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr', None)
            self.register_parameter('RMi', None)
            self.register_parameter('RVrr', None)
            self.register_parameter('RVri', None)
            self.register_parameter('RVii', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9)  # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert (xr.shape == xi.shape)
        assert (xr.size(1) == self.num_features)

    def forward(self, xr, xi):
        self._check_input_dim(xr, xi)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i != 1]
        vdim = [1] * xr.dim()
        vdim[1] = xr.size(1)
        # Mean M Computation and Centering
        # Includes running mean update if training and running.
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                tRMr = self.RMr.detach()
                tRMi = self.RMi.detach()
                tRMr.lerp_(Mr.squeeze(), exponential_average_factor)
                tRMi.lerp_(Mi.squeeze(), exponential_average_factor)
                self.RMr = tRMr
                self.RMi = tRMi
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr - Mr, xi - Mi
        # Variance Matrix V Computation
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                tRVrr = self.RVrr.detach()
                tRVri = self.RVri.detach()
                tRVii = self.RVii.detach()
                tRVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                tRVri.lerp_(Vri.squeeze(), exponential_average_factor)
                tRVii.lerp_(Vii.squeeze(), exponential_average_factor)
                self.RVrr = tRVrr
                self.RVri = tRVri
                self.RVii = tRVii
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr = Vrr + self.eps
        Vri = Vri
        Vii = Vii + self.eps
        # Matrix Inverse Square Root U = V^-0.5
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, Vri, Vri, value=-1)
        s = delta.sqrt()
        t = (tau + 2 * s).sqrt()
        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst = (s * t).reciprocal()
        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = (- Vri) * rst
        if self.affine:
            Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii
        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)
        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)
        return yr, yi

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class CLeakyReLU(nn.LeakyReLU):
    def forward(self, xr, xi):
        return F.leaky_relu(xr, self.negative_slope, self.inplace), F.leaky_relu(xi, self.negative_slope, self.inplace)


def pad2d_as(x1, x2):
    # Pad x1 to have same size with x2
    # inputs are NCHW
    diffH = x2.size()[2] - x1.size()[2]
    diffW = x2.size()[3] - x1.size()[3]
    return F.pad(x1, (0, diffW, 0, diffH))  # (L,R,T,B)


def padded_cat(x1, x2, dim):
    # NOTE: Use torch.cat with pad instead when merged
    #  > https://github.com/pytorch/pytorch/pull/11494
    x1 = pad2d_as(x1, x2)
    x1 = torch.cat([x1, x2], dim=dim)
    return x1


class DCU_Net_Encoder(nn.Module):
    def __init__(self, conv_cfg, leaky_slope):
        super(DCU_Net_Encoder, self).__init__()
        self.conv = ComplexConvWrapper(nn.Conv2d, *conv_cfg, bias=False)
        self.bn = ComplexBatchNorm(conv_cfg[1])
        self.act = CLeakyReLU(leaky_slope, inplace=True)

    def forward(self, xr, xi):
        xr, xi = self.conv(xr, xi)
        xr, xi = self.bn(xr, xi)
        xr, xi = self.act(xr, xi)
        return xr, xi


class DCU_Net_Decoder(nn.Module):
    def __init__(self, dconv_cfg, leaky_slope):
        super(DCU_Net_Decoder, self).__init__()
        self.dconv = ComplexConvWrapper(nn.ConvTranspose2d, *dconv_cfg, bias=False)
        self.bn = ComplexBatchNorm(dconv_cfg[1])
        self.act = CLeakyReLU(leaky_slope, inplace=True)

    def forward(self, xr, xi, skip=None):
        if skip is not None:
            xr, xi = padded_cat(xr, skip[0], dim=1), padded_cat(xi, skip[1], dim=1)
        xr, xi = self.dconv(xr, xi)
        xr, xi = self.bn(xr, xi)
        xr, xi = self.act(xr, xi)
        return xr, xi


class TinyDCU_Net_16(nn.Module):
    def __init__(self):
        super(TinyDCU_Net_16, self).__init__()
        self.ratio_mask_type = "BDT"
        # input [1, 1, x, 161] output [1, 8, x, 81]
        self.encoder1 = DCU_Net_Encoder([1, 8, [5, 7], [2, 2], [3, 3]], 0.1)
        # input [1, 8, x, 81] output [1, 8, x, 41]
        self.encoder2 = DCU_Net_Encoder([8, 8, [5, 7], [1, 2], [4, 3]], 0.1)
        # input [1, 8, x, 41] output [1, 8, x, 21]
        self.encoder3 = DCU_Net_Encoder([8, 8, [5, 7], [2, 2], [3, 3]], 0.1)
        # input [1, 8, x, 21] output [1, 8, x, 11]
        self.encoder4 = DCU_Net_Encoder([8, 8, [3, 5], [1, 2], [2, 2]], 0.1)
        # input [1, 8, x, 11] output [1, 16, x, 6]
        self.encoder5 = DCU_Net_Encoder([8, 16, [3, 5], [2, 2], [1, 2]], 0.1)
        # input [1, 16, x, 6] output [1, 16, x, 3]
        self.encoder6 = DCU_Net_Encoder([16, 16, [3, 5], [1, 2], [2, 2]], 0.1)
        # input [1, 16, x, 81] output [1, 16, x, 2]
        self.encoder7 = DCU_Net_Encoder([16, 16, [3, 5], [2, 2], [1, 2]], 0.1)
        # input [1, 16, x, 2] output [1, 16, x, 1]
        self.encoder8 = DCU_Net_Encoder([16, 16, [3, 5], [1, 2], [2, 2]], 0.1)
        # decoder
        self.decoder1 = DCU_Net_Decoder([16, 16, [3, 5], [1, 2], [0, 2]], 0.1)
        self.decoder2 = DCU_Net_Decoder([32, 16, [3, 5], [2, 2], [0, 2]], 0.1)
        self.decoder3 = DCU_Net_Decoder([32, 16, [3, 5], [1, 2], [0, 2]], 0.1)
        self.decoder4 = DCU_Net_Decoder([32, 8, [3, 5], [2, 2], [0, 2]], 0.1)
        self.decoder5 = DCU_Net_Decoder([16, 8, [3, 5], [1, 2], [0, 2]], 0.1)
        self.decoder6 = DCU_Net_Decoder([16, 8, [5, 7], [2, 2], [0, 3]], 0.1)
        self.decoder7 = DCU_Net_Decoder([16, 8, [5, 7], [1, 2], [0, 3]], 0.1)
        self.out = ComplexConvWrapper(nn.ConvTranspose2d, 16, 1, [5, 7], [2, 2], [0, 3], bias=True)

    def get_ratio_mask(self, outr, outi):
        def inner_fn(r, i):
            if self.ratio_mask_type == 'BDSS':
                return torch.sigmoid(outr) * r, torch.sigmoid(outi) * i
            else:
                # Polar cordinate masks
                # x1.4 slower
                mag_mask = torch.sqrt(outr ** 2 + outi ** 2)
                # M_phase = O/|O| for O = g(X)
                # Same phase rotate(theta), for phase mask O/|O| and O.
                phase_rotate = torch.atan2(outi, outr)
                if self.ratio_mask_type == 'BDT':
                    mag_mask = torch.tanh(mag_mask)
                mag = mag_mask * torch.sqrt(r ** 2 + i ** 2)
                phase = phase_rotate + torch.atan2(i, r)
                # return real, imag
                return mag * torch.cos(phase), mag * torch.sin(phase)

        return inner_fn

    def forward(self, input_real, input_imag):
        # encode part
        skips = list()
        xr, xi = self.encoder1(input_real, input_imag)
        skips.append((xr, xi))
        xr, xi = self.encoder2(xr, xi)
        skips.append((xr, xi))
        xr, xi = self.encoder3(xr, xi)
        skips.append((xr, xi))
        xr, xi = self.encoder4(xr, xi)
        skips.append((xr, xi))
        xr, xi = self.encoder5(xr, xi)
        skips.append((xr, xi))
        xr, xi = self.encoder6(xr, xi)
        skips.append((xr, xi))
        xr, xi = self.encoder7(xr, xi)
        skips.append((xr, xi))
        xr, xi = self.encoder8(xr, xi)
        # decode part
        skip = None  # First decoder input x is same as skip, drop skip.
        xr, xi = self.decoder1(xr, xi, skip)
        skip = skips.pop()
        xr, xi = self.decoder2(xr, xi, skip)
        skip = skips.pop()
        xr, xi = self.decoder3(xr, xi, skip)
        skip = skips.pop()
        xr, xi = self.decoder4(xr, xi, skip)
        skip = skips.pop()
        xr, xi = self.decoder5(xr, xi, skip)
        skip = skips.pop()
        xr, xi = self.decoder6(xr, xi, skip)
        skip = skips.pop()
        xr, xi = self.decoder7(xr, xi, skip)
        skip = skips.pop()

        xr, xi = padded_cat(xr, skip[0], dim=1), padded_cat(xi, skip[1], dim=1)
        xr, xi = self.out(xr, xi)

        xr, xi = pad2d_as(xr, input_real), pad2d_as(xi, input_imag)
        ratio_mask_fn = self.get_ratio_mask(xr, xi)
        return ratio_mask_fn(input_real, input_imag)
