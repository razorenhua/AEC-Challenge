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


class U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


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


class KiU_Net(nn.Module):

    def __init__(self, img_ch=1, output_ch=1):
        super(KiU_Net, self).__init__()

        self.EncodeConvPool1 = ConvPoolBlock(ch_in=img_ch, ch_out=64)
        self.EncodeConv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.EncodeCrossResidual1 = CrossResidualBlock(ch_in=64, ch_out=64)

        self.EncodeConvPool2 = ConvPoolBlock(ch_in=64, ch_out=128)
        self.EncodeConv2 = ConvBlock(ch_in=64, ch_out=128)
        self.EncodeCrossResidual2 = CrossResidualBlock(ch_in=128, ch_out=128)

        self.EncodeConvPool3 = ConvPoolBlock(ch_in=128, ch_out=256)
        self.EncodeConv3 = ConvBlock(ch_in=128, ch_out=256)
        self.EncodeCrossResidual3 = CrossResidualBlock(ch_in=256, ch_out=256)

        self.DecodeConv1 = ConvBlock(ch_in=256, ch_out=128)
        self.DecodeConvUp1 = ConvUpBlock(ch_in=256, ch_out=128)
        self.DecodeCrossResidual1 = CrossResidualBlock(ch_in=128, ch_out=128)

        self.DecodeConv2 = ConvBlock(ch_in=128, ch_out=64)
        self.DecodeConvUp2 = ConvUpBlock(ch_in=128, ch_out=64)
        self.DecodeCrossResidual2 = CrossResidualBlock(ch_in=64, ch_out=64)

        self.DecodeConv3 = ConvBlock(ch_in=64, ch_out=8)
        self.DecodeConvUp3 = ConvUpBlock(ch_in=64, ch_out=8)

        self.final = nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.EncodeConvPool1(x)
        out1 = self.EncodeConv1(x)
        out, out1 = self.EncodeCrossResidual1(out, out1, 2)

        u1 = out
        o1 = out1

        out = self.EncodeConvPool2(out)
        out1 = self.EncodeConv2(out1)
        out, out1 = self.EncodeCrossResidual2(out, out1, 4)

        u2 = out
        o2 = out1

        out = self.EncodeConvPool3(out)
        out1 = self.EncodeConv3(out1)
        out, out1 = self.EncodeCrossResidual3(out, out1, 8)
        # End of encoder block

        out = self.DecodeConvUp1(out)
        out1 = self.DecodeConv1(out1)
        out, out1 = self.DecodeCrossResidual1(out, out1, 4)

        out = torch.add(out, u2)
        out1 = torch.add(out1, o2)

        out = self.DecodeConvUp2(out)
        out1 = self.DecodeConv2(out1)
        out, out1 = self.DecodeCrossResidual2(out, out1, 2)

        out = torch.add(out, u1)
        out1 = torch.add(out1, o1)

        out = self.DecodeConvUp3(out)
        out1 = self.DecodeConv3(out1)

        out = torch.add(out, out1)

        out = self.final(out)
        # End of decoder block
        return out


class R2U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2AttU_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


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


class AEncoder(nn.Module):
    def __init__(self):
        super(AEncoder, self).__init__()
        self.pad1 = nn.ConstantPad2d((2, 2, 1, 0), value=0.)
        self.pad2 = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.en1 = nn.Sequential(
            nn.ConstantPad2d((2, 2, 1, 0), value=0.),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(16),
            nn.ELU())
        self.en2 = nn.Sequential(
            self.pad1,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ELU())
        self.en3 = nn.Sequential(
            self.pad1,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ELU())
        self.en4 = nn.Sequential(
            self.pad1,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ELU())
        self.en5 = nn.Sequential(
            self.pad1,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ELU())

    def forward(self, x):
        x_list = []
        x = self.en1(x)
        x_list.append(x)
        x = self.en2(x)
        x_list.append(x)
        x = self.en3(x)
        x_list.append(x)
        x = self.en4(x)
        x_list.append(x)
        x = self.en5(x)
        return x, x_list


class MiniAEncoder(nn.Module):
    def __init__(self):
        super(MiniAEncoder, self).__init__()
        self.pad1 = nn.ConstantPad2d((2, 2, 1, 0), value=0.)
        self.pad2 = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.en1 = nn.Sequential(
            nn.ConstantPad2d((2, 2, 1, 0), value=0.),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(8),
            nn.ELU())
        self.en2 = nn.Sequential(
            self.pad1,
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(16),
            nn.ELU())
        self.en3 = nn.Sequential(
            self.pad1,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(16),
            nn.ELU())
        self.en4 = nn.Sequential(
            self.pad1,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(16),
            nn.ELU())
        self.en5 = nn.Sequential(
            self.pad1,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ELU())

    def forward(self, x):
        x_list = []
        x = self.en1(x)
        x_list.append(x)
        x = self.en2(x)
        x_list.append(x)
        x = self.en3(x)
        x_list.append(x)
        x = self.en4(x)
        x_list.append(x)
        x = self.en5(x)
        return x, x_list


class ADecoder(nn.Module):
    def __init__(self):
        super(ADecoder, self).__init__()
        self.up_f = up_Chomp_F(1)
        self.down_f = down_Chomp_F(2)
        self.chomp_t = Chomp_T(1)
        self.de1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(64),
            nn.ELU())
        self.de2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64 * 2, out_channels=32, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(32),
            nn.ELU())
        self.de3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32 * 2, out_channels=32, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(32),
            nn.ELU())
        self.de4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32 * 2, out_channels=16, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(16),
            nn.ELU())
        self.de5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16 * 2, out_channels=16, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(16),
            nn.ELU())

    def forward(self, x, x_list):
        de_list = []
        x = self.de1(x)
        de_list.append(x)
        x = self.de2(torch.cat((x, x_list[-1]), dim=1))
        de_list.append(x)
        x = self.de3(torch.cat((x, x_list[-2]), dim=1))
        de_list.append(x)
        x = self.de4(torch.cat((x, x_list[-3]), dim=1))
        de_list.append(x)
        x = self.de5(torch.cat((x, x_list[-4]), dim=1))
        de_list.append(x)
        return de_list


class MiniADecoder(nn.Module):
    def __init__(self):
        super(MiniADecoder, self).__init__()
        self.up_f = up_Chomp_F(1)
        self.down_f = down_Chomp_F(2)
        self.chomp_t = Chomp_T(1)
        self.de1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(16),
            nn.ELU())
        self.de2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16 * 2, out_channels=16, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(16),
            nn.ELU())
        self.de3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16 * 2, out_channels=16, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(16),
            nn.ELU())
        self.de4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16 * 2, out_channels=8, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(8),
            nn.ELU())
        self.de5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8 * 2, out_channels=8, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(8),
            nn.ELU())

    def forward(self, x, x_list):
        de_list = []
        x = self.de1(x)
        de_list.append(x)
        x = self.de2(torch.cat((x, x_list[-1]), dim=1))
        de_list.append(x)
        x = self.de3(torch.cat((x, x_list[-2]), dim=1))
        de_list.append(x)
        x = self.de4(torch.cat((x, x_list[-3]), dim=1))
        de_list.append(x)
        x = self.de5(torch.cat((x, x_list[-4]), dim=1))
        de_list.append(x)
        return de_list


class AUnet(nn.Module):
    def __init__(self):
        super(AUnet, self).__init__()
        self.en = AEncoder()
        self.de = ADecoder()

    def forward(self, x):
        x, en_list = self.en(x)
        de_list = self.de(x, en_list)
        return de_list


class MiniAUnet(nn.Module):
    def __init__(self):
        super(MiniAUnet, self).__init__()
        self.en = MiniAEncoder()
        self.de = MiniADecoder()

    def forward(self, x):
        x, en_list = self.en(x)
        de_list = self.de(x, en_list)
        return de_list


class Attention_Block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_Block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels=F_g, out_channels=F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels=F_l, out_channels=F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels=F_int, out_channels=1, kernel_size=1, stride=1, padding=0),
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


class MEncoder(nn.Module):
    def __init__(self):
        super(MEncoder, self).__init__()
        self.pad1 = nn.ConstantPad2d((0, 0, 1, 0), value=0.)
        self.pad2 = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.pad3 = nn.ConstantPad2d((2, 2, 1, 0), value=0.)
        self.fen1 = nn.Sequential(
            self.pad3,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5), stride=(1, 1)))
        self.ben1 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ELU())
        self.fen2 = nn.Sequential(
            self.pad3,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5), stride=(1, 2)))
        self.ben2 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ELU())
        self.fen3 = nn.Sequential(
            self.pad3,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 5), stride=(1, 2)))
        self.ben3 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ELU())
        self.fen4 = nn.Sequential(
            self.pad3,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 5), stride=(1, 2)))
        self.ben4 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ELU())
        self.fen5 = nn.Sequential(
            self.pad3,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 5), stride=(1, 2)))
        self.ben5 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ELU())
        self.en6 = nn.Sequential(
            self.pad3,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ELU())
        self.point_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.Sigmoid())
        self.point_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.Sigmoid())
        self.point_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1),
            nn.Sigmoid())
        self.point_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1),
            nn.Sigmoid())
        self.point_conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x, att_list):
        x_list = []
        x = self.fen1(x)
        x = self.ben1(x * self.point_conv1(att_list[-1]))
        x = self.fen2(x)
        x = self.ben2(x * self.point_conv2(att_list[-2]))
        x_list.append(x)
        x = self.fen3(x)
        x = self.ben3(x * self.point_conv3(att_list[-3]))
        x_list.append(x)
        x = self.fen4(x)
        x = self.ben4(x * self.point_conv4(att_list[-4]))
        x_list.append(x)
        x = self.fen5(x)
        x = self.ben5(x * self.point_conv5(att_list[-5]))
        x_list.append(x)
        x = self.en6(x)
        x_list.append(x)
        return x, x_list


class MiniMEncoder(nn.Module):
    def __init__(self):
        super(MiniMEncoder, self).__init__()
        self.pad1 = nn.ConstantPad2d((0, 0, 1, 0), value=0.)
        self.pad2 = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.pad3 = nn.ConstantPad2d((2, 2, 1, 0), value=0.)
        self.fen1 = nn.Sequential(
            self.pad3,
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2, 5), stride=(1, 1)))
        self.ben1 = nn.Sequential(
            nn.BatchNorm2d(8),
            nn.ELU())
        self.fen2 = nn.Sequential(
            self.pad3,
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2, 5), stride=(1, 2)))
        self.ben2 = nn.Sequential(
            nn.BatchNorm2d(8),
            nn.ELU())
        self.fen3 = nn.Sequential(
            self.pad3,
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2, 5), stride=(1, 2)))
        self.ben3 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ELU())
        self.fen4 = nn.Sequential(
            self.pad3,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5), stride=(1, 2)))
        self.ben4 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ELU())
        self.fen5 = nn.Sequential(
            self.pad3,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5), stride=(1, 2)))
        self.ben5 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ELU())
        self.en6 = nn.Sequential(
            self.pad3,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ELU())
        self.point_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1),
            nn.Sigmoid())
        self.point_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1),
            nn.Sigmoid())
        self.point_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.Sigmoid())
        self.point_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.Sigmoid())
        self.point_conv5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x, att_list):
        x_list = []
        x = self.fen1(x)
        x = self.ben1(x * self.point_conv1(att_list[-1]))
        x = self.fen2(x)
        x = self.ben2(x * self.point_conv2(att_list[-2]))
        x_list.append(x)
        x = self.fen3(x)
        x = self.ben3(x * self.point_conv3(att_list[-3]))
        x_list.append(x)
        x = self.fen4(x)
        x = self.ben4(x * self.point_conv4(att_list[-4]))
        x_list.append(x)
        x = self.fen5(x)
        x = self.ben5(x * self.point_conv5(att_list[-5]))
        x_list.append(x)
        x = self.en6(x)
        x_list.append(x)
        return x, x_list


class MDecoder(nn.Module):
    def __init__(self):
        super(MDecoder, self).__init__()
        self.up_f = up_Chomp_F(1)
        self.down_f = down_Chomp_F(2)
        self.chomp_t = Chomp_T(1)
        self.att1 = Attention_Block(64, 64, 64)
        self.att2 = Attention_Block(64, 64, 64)
        self.att3 = Attention_Block(32, 32, 32)
        self.att4 = Attention_Block(32, 32, 32)
        self.att5 = Attention_Block(16, 16, 16)
        self.de1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64 * 2, out_channels=64, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(64),
            nn.ELU())
        self.de2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64 * 2, out_channels=32, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(32),
            nn.ELU())
        self.de3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32 * 2, out_channels=32, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(32),
            nn.ELU())
        self.de4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32 * 2, out_channels=16, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(16),
            nn.ELU())
        self.de5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16 * 2, out_channels=16, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(16),
            nn.ELU())
        self.de6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 1), stride=(1, 1)))

    def forward(self, x, en_list):
        en_list[-1] = self.att1(x, en_list[-1])
        x = self.de1(torch.cat((x, en_list[-1]), dim=1))
        en_list[-2] = self.att2(x, en_list[-2])
        x = self.de2(torch.cat((x, en_list[-2]), dim=1))
        en_list[-3] = self.att3(x, en_list[-3])
        x = self.de3(torch.cat((x, en_list[-3]), dim=1))
        en_list[-4] = self.att4(x, en_list[-4])
        x = self.de4(torch.cat((x, en_list[-4]), dim=1))
        en_list[-5] = self.att5(x, en_list[-5])
        x = self.de5(torch.cat((x, en_list[-5]), dim=1))
        x = self.de6(x)
        return x


class MiniMDecoder(nn.Module):
    def __init__(self):
        super(MiniMDecoder, self).__init__()
        self.up_f = up_Chomp_F(1)
        self.down_f = down_Chomp_F(2)
        self.chomp_t = Chomp_T(1)
        self.att1 = Attention_Block(32, 32, 32)
        self.att2 = Attention_Block(16, 16, 16)
        self.att3 = Attention_Block(16, 16, 16)
        self.att4 = Attention_Block(16, 16, 16)
        self.att5 = Attention_Block(8, 8, 8)
        self.de1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32 * 2, out_channels=16, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(16),
            nn.ELU())
        self.de2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16 * 2, out_channels=16, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(16),
            nn.ELU())
        self.de3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16 * 2, out_channels=16, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(16),
            nn.ELU())
        self.de4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16 * 2, out_channels=8, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(8),
            nn.ELU())
        self.de5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8 * 2, out_channels=8, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(8),
            nn.ELU())
        self.de6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(1, 1), stride=(1, 1)))

    def forward(self, x, en_list):
        en_list[-1] = self.att1(x, en_list[-1])
        x = self.de1(torch.cat((x, en_list[-1]), dim=1))
        en_list[-2] = self.att2(x, en_list[-2])
        x = self.de2(torch.cat((x, en_list[-2]), dim=1))
        en_list[-3] = self.att3(x, en_list[-3])
        x = self.de3(torch.cat((x, en_list[-3]), dim=1))
        en_list[-4] = self.att4(x, en_list[-4])
        x = self.de4(torch.cat((x, en_list[-4]), dim=1))
        en_list[-5] = self.att5(x, en_list[-5])
        x = self.de5(torch.cat((x, en_list[-5]), dim=1))
        x = self.de6(x)
        return x


class GLU(nn.Module):
    def __init__(self, dilation, in_channel):
        super(GLU, self).__init__()
        self.pad = nn.ConstantPad1d((dilation * 10, 0), value=0.)
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(num_features=64))
        self.left_conv = nn.Sequential(
            nn.ELU(),
            self.pad,
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11, dilation=dilation),
            nn.BatchNorm1d(num_features=64))
        self.right_conv = nn.Sequential(
            nn.ELU(),
            self.pad,
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11, dilation=dilation),
            nn.BatchNorm1d(num_features=64),
            nn.Sigmoid()
        )
        self.out_conv = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(num_features=512))
        self.out_elu = nn.ELU()

    def forward(self, inpt):
        x = inpt
        x = self.in_conv(x)
        x1 = self.left_conv(x)
        x2 = self.right_conv(x)
        x = x1 * x2
        x = self.out_conv(x)
        x = x + inpt
        x = self.out_elu(x)
        return x


class MiniGLU(nn.Module):
    def __init__(self, dilation, in_channel):
        super(MiniGLU, self).__init__()
        self.pad = nn.ConstantPad1d((dilation * 4, 0), value=0.)
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=16, kernel_size=1),
            nn.BatchNorm1d(num_features=16))
        self.left_conv = nn.Sequential(
            nn.ELU(),
            self.pad,
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, dilation=dilation),
            nn.BatchNorm1d(num_features=16))
        self.right_conv = nn.Sequential(
            nn.ELU(),
            self.pad,
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, dilation=dilation),
            nn.BatchNorm1d(num_features=16),
            nn.Sigmoid()
        )
        self.out_conv = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(num_features=256))
        self.out_elu = nn.ELU()

    def forward(self, inpt):
        x = inpt
        x = self.in_conv(x)
        x1 = self.left_conv(x)
        x2 = self.right_conv(x)
        x = x1 * x2
        x = self.out_conv(x)
        x = x + inpt
        x = self.out_elu(x)
        return x


class MNet(nn.Module):
    def __init__(self):
        super(MNet, self).__init__()
        self.en = MEncoder()
        self.de_real = MDecoder()
        self.de_imag = MDecoder()
        self.glu_list = nn.ModuleList([GLU(dilation=2 ** i, in_channel=512) for i in range(6)])

    def forward(self, x, att_list):
        x, en_list = self.en(x, att_list)
        batch_size, _, seq_len, _ = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, -1, seq_len)
        x_skip = Variable(torch.zeros(x.shape), requires_grad=True).to(x.device)
        for i in range(6):
            x = self.glu_list[i](x)
            x_skip = x_skip + x
        x = x_skip
        x = x.view(batch_size, 64, 8, seq_len)
        x = x.permute(0, 1, 3, 2).contiguous()
        x_real = self.de_real(x, en_list)
        x_imag = self.de_imag(x, en_list)
        x = torch.cat((x_real, x_imag), dim=1)
        del x_skip
        return x


class MiniMNet(nn.Module):
    def __init__(self):
        super(MiniMNet, self).__init__()
        self.en = MiniMEncoder()
        self.de_real = MiniMDecoder()
        self.de_imag = MiniMDecoder()
        self.glu_list = nn.ModuleList([MiniGLU(dilation=2 ** i, in_channel=256) for i in range(6)])

    def forward(self, x, att_list):
        x, en_list = self.en(x, att_list)
        batch_size, _, seq_len, _ = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, -1, seq_len)
        x_skip = Variable(torch.zeros(x.shape), requires_grad=True).to(x.device)
        for i in range(6):
            x = self.glu_list[i](x)
            x_skip = x_skip + x
        x = x_skip
        x = x.view(batch_size, 32, 8, seq_len)
        x = x.permute(0, 1, 3, 2).contiguous()
        x_real = self.de_real(x, en_list)
        x_imag = self.de_imag(x, en_list)
        x = torch.cat((x_real, x_imag), dim=1)
        del x_skip
        return x


class Stage_GRU(nn.Module):
    def __init__(self):
        super(Stage_GRU, self).__init__()
        self.pad = nn.ConstantPad2d((2, 2, 1, 0), value=0.)
        self.pre_conv = nn.Sequential(
            self.pad,
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(2, 5), stride=(1, 1)),
            nn.ELU())
        self.conv_xz = nn.Sequential(
            self.pad,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5)))
        self.conv_xr = nn.Sequential(
            self.pad,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5)))
        self.conv_xn = nn.Sequential(
            self.pad,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5)))
        self.conv_hz = nn.Sequential(
            self.pad,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5)))
        self.conv_hr = nn.Sequential(
            self.pad,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5)))
        self.conv_hn = nn.Sequential(
            self.pad,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5)))

    def forward(self, x, h=None):
        x = self.pre_conv(x)
        if h is None:
            z = torch.sigmoid(self.conv_xz(x))
            f = torch.tanh(self.conv_xn(x))
            h = z * f
        else:
            z = torch.sigmoid(self.conv_xz(x) + self.conv_hz(h))
            r = torch.sigmoid(self.conv_xr(x) + self.conv_hr(h))
            n = torch.tanh(self.conv_xn(x) + self.conv_hn(r * h))
            h = (1 - z) * h + z * n
        return h


class MiniStageGRU(nn.Module):
    def __init__(self):
        super(MiniStageGRU, self).__init__()
        self.pad = nn.ConstantPad2d((2, 2, 1, 0), value=0.)
        self.pre_conv = nn.Sequential(
            self.pad,
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(2, 5), stride=(1, 1)),
            nn.ELU())
        self.conv_xz = nn.Sequential(
            self.pad,
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2, 5)))
        self.conv_xr = nn.Sequential(
            self.pad,
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2, 5)))
        self.conv_xn = nn.Sequential(
            self.pad,
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2, 5)))
        self.conv_hz = nn.Sequential(
            self.pad,
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2, 5)))
        self.conv_hr = nn.Sequential(
            self.pad,
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2, 5)))
        self.conv_hn = nn.Sequential(
            self.pad,
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2, 5)))

    def forward(self, x, h=None):
        x = self.pre_conv(x)
        if h is None:
            z = torch.sigmoid(self.conv_xz(x))
            f = torch.tanh(self.conv_xn(x))
            h = z * f
        else:
            z = torch.sigmoid(self.conv_xz(x) + self.conv_hz(h))
            r = torch.sigmoid(self.conv_xr(x) + self.conv_hr(h))
            n = torch.tanh(self.conv_xn(x) + self.conv_hn(r * h))
            h = (1 - z) * h + z * n
        return h


class DARCCN(nn.Module):
    def __init__(self, t=1):
        super(DARCCN, self).__init__()
        self.aunet = AUnet()
        self.mnet = MNet()
        self.sgru = Stage_GRU()
        self.Iter = t

    def forward(self, x):
        ori_x = x
        curr_x = x
        x = torch.cat((ori_x, curr_x), dim=1)
        h = self.sgru(x, None)
        att_list = self.aunet(h)
        curr_x = self.mnet(h, att_list)
        return curr_x.squeeze(0)


class MiniDARCCN(nn.Module):
    def __init__(self, t=1):
        super(MiniDARCCN, self).__init__()
        self.aunet = MiniAUnet()
        self.mnet = MiniMNet()
        self.sgru = MiniStageGRU()
        self.Iter = t

    def forward(self, x):
        ori_x = x
        curr_x = x
        x = torch.cat((ori_x, curr_x), dim=1)
        h = self.sgru(x, None)
        att_list = self.aunet(h)
        curr_x = self.mnet(h, att_list)
        return curr_x


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


class DU_Net_Encode(nn.Module):
    def __init__(self):
        super(DU_Net_Encode, self).__init__()
        self.Conv1 = conv_block_elu(ch_in=2, ch_out=16)
        self.Conv2 = conv_block_elu(ch_in=16, ch_out=32)
        self.Conv3 = conv_block_elu(ch_in=32, ch_out=32)
        self.Conv4 = conv_block_elu(ch_in=32, ch_out=64)
        self.Conv5 = conv_block_elu(ch_in=64, ch_out=64)

    def forward(self, x):
        en_list = []
        x = self.Conv1(x)
        en_list.append(x)
        x = self.Conv2(x)
        en_list.append(x)
        x = self.Conv3(x)
        en_list.append(x)
        x = self.Conv4(x)
        en_list.append(x)
        x = self.Conv5(x)
        return x, en_list


class DU_Net_Decode(nn.Module):
    def __init__(self):
        super(DU_Net_Decode, self).__init__()
        self.upConv1 = up_block_elu(ch_in=64, ch_out=64)
        self.upConv2 = up_block_elu(ch_in=64 * 2, ch_out=32)
        self.upConv3 = up_block_elu(ch_in=32 * 2, ch_out=32)
        self.upConv4 = up_block_elu(ch_in=32 * 2, ch_out=16)
        self.upConv5 = up_block_elu(ch_in=16 * 2, ch_out=16)
        self.de = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x, en_list):
        x = self.upConv1(x)
        x = self.upConv2(torch.cat((x, en_list[-1]), dim=1))
        x = self.upConv3(torch.cat((x, en_list[-2]), dim=1))
        x = self.upConv4(torch.cat((x, en_list[-3]), dim=1))
        x = self.upConv5(torch.cat((x, en_list[-4]), dim=1))
        x = self.de(x)
        return x


class DU_Net(nn.Module):
    def __init__(self):
        super(DU_Net, self).__init__()

        self.encode = DU_Net_Encode()

        self.real_decode = DU_Net_Decode()
        self.imag_decode = DU_Net_Decode()

    def forward(self, x):
        # encoding path
        x, en_list = self.encode(x)
        x_real = self.real_decode(x, en_list)
        x_imag = self.imag_decode(x, en_list)
        x = torch.cat((x_real, x_imag), dim=1)
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


class DCU_Net_16(nn.Module):
    def __init__(self):
        super(DCU_Net_16, self).__init__()
        self.ratio_mask_type = "BDT"
        # encoder
        self.encoder1 = DCU_Net_Encoder([1, 32, [5, 7], [2, 2], [2, 3]], 0.1)
        self.encoder2 = DCU_Net_Encoder([32, 32, [5, 7], [1, 2], [2, 3]], 0.1)
        self.encoder3 = DCU_Net_Encoder([32, 64, [5, 7], [2, 2], [2, 3]], 0.1)
        self.encoder4 = DCU_Net_Encoder([64, 64, [3, 5], [1, 2], [1, 2]], 0.1)
        self.encoder5 = DCU_Net_Encoder([64, 64, [3, 5], [2, 2], [1, 2]], 0.1)
        self.encoder6 = DCU_Net_Encoder([64, 64, [3, 5], [1, 2], [1, 2]], 0.1)
        self.encoder7 = DCU_Net_Encoder([64, 64, [3, 5], [2, 2], [1, 2]], 0.1)
        self.encoder8 = DCU_Net_Encoder([64, 64, [3, 5], [1, 2], [1, 2]], 0.1)
        # decoder
        self.decoder1 = DCU_Net_Decoder([64, 64, [3, 5], [1, 2], [1, 2]], 0.1)
        self.decoder2 = DCU_Net_Decoder([128, 64, [3, 5], [2, 2], [1, 2]], 0.1)
        self.decoder3 = DCU_Net_Decoder([128, 64, [3, 5], [1, 2], [1, 2]], 0.1)
        self.decoder4 = DCU_Net_Decoder([128, 64, [3, 5], [2, 2], [1, 2]], 0.1)
        self.decoder5 = DCU_Net_Decoder([128, 64, [5, 7], [1, 2], [1, 2]], 0.1)
        self.decoder6 = DCU_Net_Decoder([128, 32, [5, 7], [2, 2], [2, 3]], 0.1)
        self.decoder7 = DCU_Net_Decoder([64, 32, [5, 7], [1, 2], [2, 3]], 0.1)
        self.out = ComplexConvWrapper(nn.ConvTranspose2d, 64, 1, [5, 7], [2, 2], [2, 3], bias=True)

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


class MiniDCU_Net_16(nn.Module):
    def __init__(self):
        super(MiniDCU_Net_16, self).__init__()
        self.ratio_mask_type = "BDT"
        self.encoder1 = DCU_Net_Encoder([1, 8, [5, 7], [2, 2], [2, 3]], 0.1)
        self.encoder2 = DCU_Net_Encoder([8, 8, [5, 7], [1, 2], [2, 3]], 0.1)
        self.encoder3 = DCU_Net_Encoder([8, 16, [5, 7], [2, 2], [2, 3]], 0.1)
        self.encoder4 = DCU_Net_Encoder([16, 16, [3, 5], [1, 2], [1, 2]], 0.1)
        self.encoder5 = DCU_Net_Encoder([16, 16, [3, 5], [2, 2], [1, 2]], 0.1)
        self.encoder6 = DCU_Net_Encoder([16, 16, [3, 5], [1, 2], [1, 2]], 0.1)
        self.encoder7 = DCU_Net_Encoder([16, 16, [3, 5], [2, 2], [1, 2]], 0.1)
        self.encoder8 = DCU_Net_Encoder([16, 32, [3, 5], [1, 2], [1, 2]], 0.1)
        # decoder
        self.decoder1 = DCU_Net_Decoder([32, 16, [3, 5], [1, 2], [1, 2]], 0.1)
        self.decoder2 = DCU_Net_Decoder([32, 16, [3, 5], [2, 2], [1, 2]], 0.1)
        self.decoder3 = DCU_Net_Decoder([32, 16, [3, 5], [1, 2], [1, 2]], 0.1)
        self.decoder4 = DCU_Net_Decoder([32, 16, [3, 5], [2, 2], [1, 2]], 0.1)
        self.decoder5 = DCU_Net_Decoder([32, 16, [5, 7], [1, 2], [1, 2]], 0.1)
        self.decoder6 = DCU_Net_Decoder([32, 8, [5, 7], [2, 2], [2, 3]], 0.1)
        self.decoder7 = DCU_Net_Decoder([16, 8, [5, 7], [1, 2], [2, 3]], 0.1)
        self.out = ComplexConvWrapper(nn.ConvTranspose2d, 16, 1, [5, 7], [2, 2], [2, 3], bias=True)

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


class MicroDCU_Net_16(nn.Module):
    def __init__(self):
        super(MicroDCU_Net_16, self).__init__()
        self.ratio_mask_type = "BDT"
        self.encoder1 = DCU_Net_Encoder([1, 2, [5, 7], [2, 2], [3, 3]], 0.1)
        self.encoder2 = DCU_Net_Encoder([2, 4, [5, 7], [1, 2], [4, 3]], 0.1)
        self.encoder3 = DCU_Net_Encoder([4, 4, [5, 7], [2, 2], [3, 3]], 0.1)
        self.encoder4 = DCU_Net_Encoder([4, 4, [3, 5], [1, 2], [2, 2]], 0.1)
        self.encoder5 = DCU_Net_Encoder([4, 8, [3, 5], [2, 2], [1, 2]], 0.1)
        self.encoder6 = DCU_Net_Encoder([8, 8, [3, 5], [1, 2], [2, 2]], 0.1)
        self.encoder7 = DCU_Net_Encoder([8, 16, [3, 5], [2, 2], [1, 2]], 0.1)
        self.encoder8 = DCU_Net_Encoder([16, 32, [3, 5], [1, 2], [2, 2]], 0.1)
        # decoder
        self.decoder1 = DCU_Net_Decoder([32, 16, [3, 5], [1, 2], [0, 2]], 0.1)
        self.decoder2 = DCU_Net_Decoder([32, 8, [3, 5], [2, 2], [0, 2]], 0.1)
        self.decoder3 = DCU_Net_Decoder([16, 8, [3, 5], [1, 2], [0, 2]], 0.1)
        self.decoder4 = DCU_Net_Decoder([16, 4, [3, 5], [2, 2], [0, 2]], 0.1)
        self.decoder5 = DCU_Net_Decoder([8, 4, [3, 5], [1, 2], [0, 2]], 0.1)
        self.decoder6 = DCU_Net_Decoder([8, 4, [5, 7], [2, 2], [0, 3]], 0.1)
        self.decoder7 = DCU_Net_Decoder([8, 2, [5, 7], [1, 2], [0, 3]], 0.1)
        self.out = ComplexConvWrapper(nn.ConvTranspose2d, 4, 1, [5, 7], [2, 2], [0, 3], bias=True)

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


class NDCU_Net_16(nn.Module):
    def __init__(self):
        super(NDCU_Net_16, self).__init__()
        self.ratio_mask_type = "BDT"
        self.encoder1 = DCU_Net_Encoder([1, 32, [5, 7], [2, 2], [3, 3]], 0.1)
        self.encoder2 = DCU_Net_Encoder([32, 32, [5, 7], [1, 2], [4, 3]], 0.1)
        self.encoder3 = DCU_Net_Encoder([32, 64, [5, 7], [2, 2], [3, 3]], 0.1)
        self.encoder4 = DCU_Net_Encoder([64, 64, [3, 5], [1, 2], [2, 2]], 0.1)
        self.encoder5 = DCU_Net_Encoder([64, 64, [3, 5], [2, 2], [1, 2]], 0.1)
        self.encoder6 = DCU_Net_Encoder([64, 64, [3, 5], [1, 2], [2, 2]], 0.1)
        self.encoder7 = DCU_Net_Encoder([64, 64, [3, 5], [2, 2], [1, 2]], 0.1)
        self.encoder8 = DCU_Net_Encoder([64, 64, [3, 5], [1, 2], [2, 2]], 0.1)
        # decoder
        self.decoder1 = DCU_Net_Decoder([64, 64, [3, 5], [1, 2], [0, 2]], 0.1)
        self.decoder2 = DCU_Net_Decoder([128, 64, [3, 5], [2, 2], [0, 2]], 0.1)
        self.decoder3 = DCU_Net_Decoder([128, 64, [3, 5], [1, 2], [0, 2]], 0.1)
        self.decoder4 = DCU_Net_Decoder([128, 64, [3, 5], [2, 2], [0, 2]], 0.1)
        self.decoder5 = DCU_Net_Decoder([128, 64, [3, 5], [1, 2], [0, 2]], 0.1)
        self.decoder6 = DCU_Net_Decoder([128, 32, [5, 7], [2, 2], [0, 3]], 0.1)
        self.decoder7 = DCU_Net_Decoder([64, 32, [5, 7], [1, 2], [0, 3]], 0.1)
        self.out = ComplexConvWrapper(nn.ConvTranspose2d, 64, 1, [5, 7], [2, 2], [0, 3], bias=True)

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


class GRU_Net_Encoder(nn.Module):
    def __init__(self, input_channel=1, input_dim=1):
        super(GRU_Net_Encoder, self).__init__()
        seq_len = input_channel * input_dim
        self.grur = torch.nn.GRU(seq_len, seq_len)
        self.grui = torch.nn.GRU(seq_len, seq_len)

    def forward(self, xr, xi):
        batch_size = xr.shape[0]
        channel = xr.shape[1]
        tsize = xr.shape[2]
        fsize = xr.shape[3]
        seq_len = channel * fsize
        xr = xr.permute(0, 2, 1, 3).contiguous()
        xr = xr.reshape(batch_size, tsize, seq_len)
        xr, hr = self.grur(xr)
        xr = xr.reshape(batch_size, tsize, channel, fsize)
        xr = xr.permute(0, 2, 1, 3).contiguous()

        xi = xi.permute(0, 2, 1, 3).contiguous()
        xi = xi.reshape(batch_size, tsize, seq_len)
        xi, hi = self.grui(xi)
        xi = xi.reshape(batch_size, tsize, channel, fsize)
        xi = xi.permute(0, 2, 1, 3).contiguous()

        return xr, xi


class DCGRU_Net_16(nn.Module):
    def __init__(self):
        super(DCGRU_Net_16, self).__init__()
        self.ratio_mask_type = "BDT"
        # input [1, 1, x, 257] output [1, 2, x, 129]
        self.cnn_encoder1 = DCU_Net_Encoder([1, 2, [5, 7], [1, 2], [2, 3]], 0.1)
        self.rnn_encoder1 = GRU_Net_Encoder(2, 129)
        # input [1, 2, x, 129] output [1, 4, x, 65]
        self.cnn_encoder2 = DCU_Net_Encoder([2, 4, [5, 7], [1, 2], [2, 3]], 0.1)
        self.rnn_encoder2 = GRU_Net_Encoder(4, 65)
        # input [1, 4, x, 65] output [1, 8, x, 33]
        self.cnn_encoder3 = DCU_Net_Encoder([4, 8, [5, 7], [1, 2], [2, 3]], 0.1)
        self.rnn_encoder3 = GRU_Net_Encoder(8, 33)
        # input [1, 8, x, 33] output [1, 16, x, 17]
        self.cnn_encoder4 = DCU_Net_Encoder([8, 16, [3, 5], [1, 2], [1, 2]], 0.1)
        self.rnn_encoder4 = GRU_Net_Encoder(16, 17)
        # input [1, 16, x, 17] output [1, 32, x, 9]
        self.cnn_encoder5 = DCU_Net_Encoder([16, 32, [3, 5], [1, 2], [1, 2]], 0.1)
        self.rnn_encoder5 = GRU_Net_Encoder(32, 9)
        # input [1, 32, x, 9] output [1, 32, x, 5]
        self.cnn_encoder6 = DCU_Net_Encoder([32, 32, [3, 5], [1, 2], [1, 2]], 0.1)
        self.rnn_encoder6 = GRU_Net_Encoder(32, 5)
        # input [1, 32, x, 5] output [1, 32, x, 3]
        self.cnn_encoder7 = DCU_Net_Encoder([32, 32, [3, 5], [1, 2], [1, 2]], 0.1)
        self.rnn_encoder7 = GRU_Net_Encoder(32, 3)
        # input [1, 32, x, 3] output [1, 64, x, 3]
        self.cnn_encoder8 = DCU_Net_Encoder([32, 64, [3, 5], [1, 1], [1, 2]], 0.1)

        # decoder
        self.cnn_decoder1 = DCU_Net_Decoder([64, 32, [3, 5], [1, 1], [0, 2]], 0.1)
        self.cnn_decoder2 = DCU_Net_Decoder([64, 32, [3, 5], [1, 2], [0, 2]], 0.1)
        self.cnn_decoder3 = DCU_Net_Decoder([64, 32, [3, 5], [1, 2], [0, 2]], 0.1)
        self.cnn_decoder4 = DCU_Net_Decoder([64, 16, [3, 5], [1, 2], [0, 2]], 0.1)
        self.cnn_decoder5 = DCU_Net_Decoder([32, 8, [3, 5], [1, 2], [0, 2]], 0.1)
        self.cnn_decoder6 = DCU_Net_Decoder([16, 4, [5, 7], [1, 2], [0, 3]], 0.1)
        self.cnn_decoder7 = DCU_Net_Decoder([8, 2, [5, 7], [1, 2], [0, 3]], 0.1)
        self.out = ComplexConvWrapper(nn.ConvTranspose2d, 4, 1, [5, 7], [1, 2], [0, 3], bias=True)

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
        xr, xi = self.cnn_encoder1(input_real, input_imag)
        hr, hi = self.rnn_encoder1(xr, xi)
        skips.append((hr, hi))
        xr, xi = self.cnn_encoder2(xr, xi)
        hr, hi = self.rnn_encoder2(xr, xi)
        skips.append((hr, hi))
        xr, xi = self.cnn_encoder3(xr, xi)
        hr, hi = self.rnn_encoder3(xr, xi)
        skips.append((hr, hi))
        xr, xi = self.cnn_encoder4(xr, xi)
        hr, hi = self.rnn_encoder4(xr, xi)
        skips.append((hr, hi))
        xr, xi = self.cnn_encoder5(xr, xi)
        hr, hi = self.rnn_encoder5(xr, xi)
        skips.append((hr, hi))
        xr, xi = self.cnn_encoder6(xr, xi)
        hr, hi = self.rnn_encoder6(xr, xi)
        skips.append((hr, hi))
        xr, xi = self.cnn_encoder7(xr, xi)
        hr, hi = self.rnn_encoder7(xr, xi)
        skips.append((hr, hi))
        xr, xi = self.cnn_encoder8(xr, xi)
        # decode part
        skip = None  # First decoder input x is same as skip, drop skip.
        xr, xi = self.cnn_decoder1(xr, xi, skip)
        skip = skips.pop()
        xr, xi = self.cnn_decoder2(xr, xi, skip)
        skip = skips.pop()
        xr, xi = self.cnn_decoder3(xr, xi, skip)
        skip = skips.pop()
        xr, xi = self.cnn_decoder4(xr, xi, skip)
        skip = skips.pop()
        xr, xi = self.cnn_decoder5(xr, xi, skip)
        skip = skips.pop()
        xr, xi = self.cnn_decoder6(xr, xi, skip)
        skip = skips.pop()
        xr, xi = self.cnn_decoder7(xr, xi, skip)
        skip = skips.pop()

        xr, xi = padded_cat(xr, skip[0], dim=1), padded_cat(xi, skip[1], dim=1)
        xr, xi = self.out(xr, xi)

        xr, xi = pad2d_as(xr, input_real), pad2d_as(xi, input_imag)
        ratio_mask_fn = self.get_ratio_mask(xr, xi)
        return ratio_mask_fn(input_real, input_imag)


class DCGRU_Net_22(nn.Module):
    def __init__(self):
        super(DCGRU_Net_22, self).__init__()
        self.ratio_mask_type = "BDT"
        # input [1, 1, x, 257] output [1, 2, x, 129]
        self.cnn_encoder1 = DCU_Net_Encoder([1, 2, [5, 7], [1, 2], [2, 3]], 0.1)
        self.rnn_encoder1 = GRU_Net_Encoder(2, 129)
        self.cnn_encoder11 = DCU_Net_Encoder([2, 2, [5, 7], [1, 1], [2, 3]], 0.1)
        # input [1, 2, x, 129] output [1, 4, x, 65]
        self.cnn_encoder2 = DCU_Net_Encoder([2, 4, [5, 7], [1, 2], [2, 3]], 0.1)
        self.rnn_encoder2 = GRU_Net_Encoder(4, 65)
        self.cnn_encoder22 = DCU_Net_Encoder([4, 4, [5, 7], [1, 1], [2, 3]], 0.1)
        # input [1, 4, x, 65] output [1, 8, x, 33]
        self.cnn_encoder3 = DCU_Net_Encoder([4, 8, [5, 7], [1, 2], [2, 3]], 0.1)
        self.rnn_encoder3 = GRU_Net_Encoder(8, 33)
        self.cnn_encoder33 = DCU_Net_Encoder([8, 8, [5, 7], [1, 1], [2, 3]], 0.1)
        # input [1, 8, x, 33] output [1, 16, x, 17]
        self.cnn_encoder4 = DCU_Net_Encoder([8, 16, [3, 5], [1, 2], [1, 2]], 0.1)
        self.rnn_encoder4 = GRU_Net_Encoder(16, 17)
        self.cnn_encoder44 = DCU_Net_Encoder([16, 16, [5, 7], [1, 1], [2, 3]], 0.1)
        # input [1, 16, x, 17] output [1, 32, x, 9]
        self.cnn_encoder5 = DCU_Net_Encoder([16, 32, [3, 5], [1, 2], [1, 2]], 0.1)
        self.rnn_encoder5 = GRU_Net_Encoder(32, 9)
        self.cnn_encoder55 = DCU_Net_Encoder([32, 32, [5, 7], [1, 1], [2, 3]], 0.1)
        # input [1, 32, x, 9] output [1, 32, x, 5]
        self.cnn_encoder6 = DCU_Net_Encoder([32, 32, [3, 5], [1, 2], [1, 2]], 0.1)
        self.rnn_encoder6 = GRU_Net_Encoder(32, 5)
        self.cnn_encoder66 = DCU_Net_Encoder([32, 32, [5, 7], [1, 1], [2, 3]], 0.1)
        # input [1, 32, x, 5] output [1, 32, x, 3]
        self.cnn_encoder7 = DCU_Net_Encoder([32, 32, [3, 5], [1, 2], [1, 2]], 0.1)
        self.rnn_encoder7 = GRU_Net_Encoder(32, 3)
        # input [1, 32, x, 3] output [1, 64, x, 3]
        self.cnn_encoder8 = DCU_Net_Encoder([32, 64, [3, 5], [1, 1], [1, 2]], 0.1)

        # decoder
        self.cnn_decoder1 = DCU_Net_Decoder([64, 32, [3, 5], [1, 1], [0, 2]], 0.1)
        self.cnn_decoder2 = DCU_Net_Decoder([64, 32, [3, 5], [1, 2], [0, 2]], 0.1)
        self.cnn_decoder3 = DCU_Net_Decoder([64, 32, [3, 5], [1, 2], [0, 2]], 0.1)
        self.cnn_decoder4 = DCU_Net_Decoder([64, 16, [3, 5], [1, 2], [0, 2]], 0.1)
        self.cnn_decoder5 = DCU_Net_Decoder([32, 8, [3, 5], [1, 2], [0, 2]], 0.1)
        self.cnn_decoder6 = DCU_Net_Decoder([16, 4, [5, 7], [1, 2], [0, 3]], 0.1)
        self.cnn_decoder7 = DCU_Net_Decoder([8, 2, [5, 7], [1, 2], [0, 3]], 0.1)
        self.out = ComplexConvWrapper(nn.ConvTranspose2d, 4, 1, [5, 7], [1, 2], [0, 3], bias=True)

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
        xr, xi = input_real, input_imag
        skips = list()
        xr, xi = self.cnn_encoder1(xr, xi)
        hr, hi = self.rnn_encoder1(xr, xi)
        xr, xi = self.cnn_encoder11(xr, xi)
        skips.append((hr, hi))
        xr, xi = self.cnn_encoder2(xr, xi)
        hr, hi = self.rnn_encoder2(xr, xi)
        xr, xi = self.cnn_encoder22(xr, xi)
        skips.append((hr, hi))
        xr, xi = self.cnn_encoder3(xr, xi)
        hr, hi = self.rnn_encoder3(xr, xi)
        xr, xi = self.cnn_encoder33(xr, xi)
        skips.append((hr, hi))
        xr, xi = self.cnn_encoder4(xr, xi)
        hr, hi = self.rnn_encoder4(xr, xi)
        xr, xi = self.cnn_encoder44(xr, xi)
        skips.append((hr, hi))
        xr, xi = self.cnn_encoder5(xr, xi)
        hr, hi = self.rnn_encoder5(xr, xi)
        xr, xi = self.cnn_encoder55(xr, xi)
        skips.append((hr, hi))
        xr, xi = self.cnn_encoder6(xr, xi)
        hr, hi = self.rnn_encoder6(xr, xi)
        xr, xi = self.cnn_encoder66(xr, xi)
        skips.append((hr, hi))
        xr, xi = self.cnn_encoder7(xr, xi)
        hr, hi = self.rnn_encoder7(xr, xi)
        skips.append((hr, hi))
        xr, xi = self.cnn_encoder8(xr, xi)
        # decode part
        skip = None  # First decoder input x is same as skip, drop skip.
        xr, xi = self.cnn_decoder1(xr, xi, skip)
        skip = skips.pop()
        xr, xi = self.cnn_decoder2(xr, xi, skip)
        skip = skips.pop()
        xr, xi = self.cnn_decoder3(xr, xi, skip)
        skip = skips.pop()
        xr, xi = self.cnn_decoder4(xr, xi, skip)
        skip = skips.pop()
        xr, xi = self.cnn_decoder5(xr, xi, skip)
        skip = skips.pop()
        xr, xi = self.cnn_decoder6(xr, xi, skip)
        skip = skips.pop()
        xr, xi = self.cnn_decoder7(xr, xi, skip)
        skip = skips.pop()

        xr, xi = padded_cat(xr, skip[0], dim=1), padded_cat(xi, skip[1], dim=1)
        xr, xi = self.out(xr, xi)

        xr, xi = pad2d_as(xr, input_real), pad2d_as(xi, input_imag)
        ratio_mask_fn = self.get_ratio_mask(xr, xi)
        return ratio_mask_fn(input_real, input_imag)
