#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torch.utils.data
from torch.autograd import Variable
from utils.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
# from lib.nn import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d



### Generator
class RDUNet_LSTM_dilations(nn.Module):

    def __init__(self, inx_chs=3, inxseg_chs=0, out_chs=3, t=0, nFeats=64, num_blk=3,
                 nDlayer=8, grRate=16, sn=True, use_lstm=True, dilations=False):
        super(RDUNet_LSTM_dilations, self).__init__()
        self.num_blk = num_blk
        self.use_lstm = use_lstm
        self.inxseg_chs = inxseg_chs
        self.dilations = dilations
        self.inx = nn.Sequential(
            nn.Conv2d(inx_chs, nFeats, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(nFeats, nFeats, kernel_size=4, stride=2, padding=1), )
        nFeats_ = nFeats
        if t>0:
            self.inxt = nn.Sequential(
                nn.Conv2d(inx_chs, int(nFeats/8), kernel_size=3, stride=1, padding=1),
                nn.Conv2d(int(nFeats/8), int(nFeats/8), kernel_size=4, stride=2, padding=1), )
            nFeats_ += t*int(nFeats/8)
        if self.inxseg_chs:
            self.inx2seg = nn.Sequential(
                nn.Conv2d(inxseg_chs, int(nFeats/8), kernel_size=3, stride=1, padding=1),
                nn.Conv2d(int(nFeats/8), int(nFeats/8), kernel_size=4, stride=2, padding=1), )
            nFeats_ += 1*int(nFeats/8)

        self.fea = nn.Sequential(
            nn.Conv2d(nFeats_, nFeats, 3, 1, 1),
            nn.ReLU()
        )

        if self.use_lstm:
            self.add_feats = nFeats
            self.lstm = lstm(nChannels=nFeats, add_feats=self.add_feats)

        modules = []
        for i in range(self.num_blk):
            modules.append(main_block(in_nc=nFeats, out_nc=nFeats,
                                      nDlayer=nDlayer, grRate=grRate, sn=sn))
        self.main_blk = nn.Sequential(*modules)
        if dilations:
            self.assp = aspp(inc=nFeats, feat=8, outc=int(nFeats/8), dilations=[1, 6, 12, 18])
            self.conv_1 = nn.Conv2d((nFeats + int(nFeats/8)) * self.num_blk, nFeats, kernel_size=1,
                                    stride=1, padding=0)
        else:
            self.conv_1 = nn.Conv2d(nFeats * self.num_blk, nFeats, kernel_size=1,
                                    stride=1, padding=0)


        self.out = nn.Sequential(
            nn.ConvTranspose2d(nFeats, nFeats, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(nFeats, out_chs, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, xt=None, seg=None,):
        # x: rain image, xt: estimate rain image of each rescusive, seg: segment image
        fea_x = self.inx(x)
        fea = []
        fea.append(fea_x)
        if xt is not None:
            fea_xt = []
            for i in range(len(xt)):
                fea_xt.append(self.inxt(xt[i]))
            fea.append(torch.cat(fea_xt, 1))

        if self.inxseg_chs:
            fea.append(self.inx2seg(seg))
        fea = self.fea(torch.cat(fea, 1))

        if self.use_lstm:
            batch_size, row, col = fea.size(0), fea.size(2), fea.size(3)
            h = Variable(torch.zeros(batch_size, self.add_feats, row, col))
            c = Variable(torch.zeros(batch_size, self.add_feats, row, col))

        outputlist = []
        for main in self.main_blk:
            if self.use_lstm:
                lstm, h, c = self.lstm(fea, h, c)
                fea = main(lstm)
            else:
                fea = main(fea)
            outputlist.append(fea)
            if self.dilations:
                assp = self.assp(fea)
                outputlist.append(assp)
        concat = torch.cat(outputlist, 1)
        fea = fea_x + self.conv_1(concat)
        out = self.out(fea) + x
        return out

class RDUNet_LSTM(nn.Module):

    def __init__(self, inx0_chs=3, inx1_chs=3, inx2_chs=150, out_chs=3, nFeats=32, num_blk=3,
                 nDlayer=8, grRate=16, sn=True, use_lstm=True, dilations=[1, 6, 12, 18]):
        super(RDUNet_LSTM, self).__init__()
        self.num_blk = num_blk
        self.use_lstm = use_lstm

        self.inx0 = nn.Sequential(
            nn.Conv2d(inx0_chs, nFeats, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(nFeats, nFeats, kernel_size=4, stride=2, padding=1), )
        if self.use_lstm:
            self.inx1 = nn.Sequential(
                nn.Conv2d(inx1_chs, nFeats, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(nFeats, nFeats, kernel_size=4, stride=2, padding=1), )
            self.inx2 = nn.Sequential(
                nn.Conv2d(inx2_chs, nFeats, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(nFeats, nFeats, kernel_size=4, stride=2, padding=1),)
            self.fea = nn.Sequential(
                nn.Conv2d(nFeats * 3, nFeats, 3, 1, 1),
                nn.ReLU()
            )
            self.add_feats = nFeats
            self.lstm = lstm(nChannels=nFeats, add_feats=self.add_feats)

            # nn.LSTM()

        modules = []
        for i in range(self.num_blk):
            modules.append(main_block(in_nc=nFeats, out_nc=nFeats,
                                      nDlayer=nDlayer, grRate=grRate, sn=sn))
        self.main_blk = nn.Sequential(*modules)
        self.assp = aspp(inc=nFeats, feat=32,outc=nFeats, dilations=dilations)
        self.conv_1 = nn.Conv2d((nFeats + nFeats) * (self.num_blk), nFeats, kernel_size=1,
                                stride=1, padding=0)

        self.out = nn.Sequential(
            nn.ConvTranspose2d(nFeats, nFeats, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(nFeats, out_chs, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x0, x1=None, x2=None):
        # x0: rain image, x1: estimate rain image, x2: segment image
        outputlist = []
        fea_x0 = self.inx0(x0)
        fea = fea_x0

        if self.use_lstm:
            fea_x1 = self.inx1(x1)
            fea_x2 = self.inx2(x2)
            batch_size, row, col = fea_x1.size(0), fea_x1.size(2), fea_x1.size(3)
            h = Variable(torch.zeros(batch_size, self.add_feats, row, col))
            c = Variable(torch.zeros(batch_size, self.add_feats, row, col))

            fea = self.fea(torch.cat((fea_x0, fea_x1, fea_x2),1))

        for main in self.main_blk:
            if self.use_lstm:
                lstm, h, c = self.lstm(fea, h, c)
                fea = main(lstm)
            else:
                fea = main(fea)
            assp = self.assp(fea)
            outputlist.append(fea)
            outputlist.append(assp)
        concat = torch.cat(outputlist, 1)
        fea = fea_x0 + self.conv_1(concat)
        out = self.out(fea) + x0
        return out



### Discriminator
class Discr_LSTM(torch.nn.Module):
    def __init__(self, outD=0, inx_chs=3, inxseg_chs=150, t=0, nFeats=32,
                 use_lstm=True, image_size=224, num_lstm=2):
        super(Discr_LSTM, self).__init__()

        self.outD = outD
        self.nfeats = nFeats
        self.image_size = image_size
        self.use_lstm = use_lstm
        self.num_lstm = num_lstm
        self.inxseg_chs = inxseg_chs

        self.inx = nn.Sequential(
            nn.Conv2d(inx_chs, nFeats, kernel_size=3, stride=1, padding=1),)
        n = 1
        if t > 0:
            self.inxt = nn.Sequential(
                nn.Conv2d(inx_chs, nFeats, kernel_size=3, stride=1, padding=1),)
            n += t
        if self.inxseg_chs:
            self.inx2seg = nn.Sequential(
                nn.Conv2d(inxseg_chs, nFeats, kernel_size=3, stride=1, padding=1),)
                # nn.Conv2d(nFeats, nFeats, kernel_size=4, stride=2, padding=1), )
            n += 1
        self.fea = nn.Sequential(
            nn.Conv2d(nFeats * n, nFeats, 3, 1, 1),
            nn.ReLU()
        )

        if self.use_lstm:
            self.add_feats = nFeats
            self.lstm = lstm(nChannels=nFeats, add_feats=self.add_feats)


        if self.outD > 0:
            self.dense = torch.nn.Linear(nFeats*8 * int((image_size/8)**2), outD**2)
        else:
            self.dense = torch.nn.Linear(nFeats*8 * int((image_size/8)**2), 1)
        self.inx = nn.Conv2d(inx_chs, nFeats, kernel_size=3, stride=1, padding=1)

        model = [
            torch.nn.LeakyReLU(0.1, inplace=True),

            spectral_norm(torch.nn.Conv2d(nFeats, nFeats, kernel_size=4, stride=2, padding=1, bias=True)),
            torch.nn.LeakyReLU(0.1, inplace=True),

            spectral_norm(torch.nn.Conv2d(nFeats, nFeats*2, kernel_size=3, stride=1, padding=1, bias=True)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(torch.nn.Conv2d(nFeats*2, nFeats*2, kernel_size=4, stride=2, padding=1, bias=True)),
            torch.nn.LeakyReLU(0.1, inplace=True),

            spectral_norm(torch.nn.Conv2d(nFeats*2, nFeats*4, kernel_size=3, stride=1, padding=1, bias=True)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(torch.nn.Conv2d(nFeats*4, nFeats*4, kernel_size=4, stride=2, padding=1, bias=True)),
            torch.nn.LeakyReLU(0.1, inplace=True),

            spectral_norm(torch.nn.Conv2d(nFeats*4, nFeats*8, kernel_size=3, stride=1, padding=1, bias=True)),
            torch.nn.LeakyReLU(0.1, inplace=True)]

        self.model = torch.nn.Sequential(*model)

    def forward(self, x, xt=None, seg=None,):
        # x: rain image, xt: estimate rain image of each rescusive, seg: segment image
        fea_x = self.inx(x)
        fea = []
        fea.append(fea_x)
        if xt is not None:
            fea_xt = []
            for i in range(len(xt)):
                fea_xt.append(self.inxt(xt[i]))
            fea.append(torch.cat(fea_xt, 1))

        if self.inxseg_chs:
            fea.append(self.inx2seg(seg))
        fea = self.fea(torch.cat(fea, 1))

        if self.use_lstm:
            batch_size, row, col = fea.size(0), fea.size(2), fea.size(3)
            h = Variable(torch.zeros(batch_size, self.add_feats, row, col))
            c = Variable(torch.zeros(batch_size, self.add_feats, row, col))
            for i in range(self.num_lstm):
                fea, h, c = self.lstm(fea, h, c)
            out = fea
        else:
            out = fea

        if self.outD > 0:
            out = self.dense(self.model(out).view(-1, self.nfeats*8 * int(((self.image_size) / 8) ** 2))).view(-1, self.outD, self.outD)

        else:
            out = self.dense(self.model(out).view(-1, self.nfeats*8 * int(((self.image_size) / 8) ** 2))).view(-1)
        return out



### sub class
class main_block(nn.Module):
    def __init__(self, in_nc=64, out_nc=64, nDlayer=4, grRate=16, sn=True):
        super(main_block, self).__init__()

        self.inc = nn.Sequential(
            # single_conv(in_nc, in_nc, sn=sn),
            rdbsn(in_nc, nDlayer, grRate, sn=sn),
        )
        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(in_nc, in_nc * 2),
            rdbsn(in_nc * 2, nDlayer, grRate, sn=sn),
        )
        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(in_nc * 2, in_nc * 4),
            rdbsn(in_nc * 4, nDlayer, grRate, sn=sn),
        )
        self.up1 = up(in_nc * 4)
        self.conv3 = nn.Sequential(
            rdbsn(in_nc * 2, nDlayer, grRate, sn=sn),
        )
        self.up2 = up(in_nc * 2)
        self.conv4 = nn.Sequential(
            rdbsn(in_nc, nDlayer, grRate, sn=sn),
        )
        self.outc = outconv(in_nc, out_nc)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch, sn=False):
        super(single_conv, self).__init__()
        if sn:
            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1)),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # x is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x2 + x1
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, sn=False):
        super(outconv, self).__init__()
        if sn:
            self.conv = spectral_norm(nn.Conv2d(in_ch, out_ch, 1))
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3, sn=False):
        super(make_dense, self).__init__()
        if sn:
            self.conv = spectral_norm(
                nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                          bias=False))
        else:
            self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                  bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class rdbsn(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, sn=False):
        super(rdbsn, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate, sn=sn))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):

        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class lstm(nn.Module):
    def __init__(self, nChannels, add_feats):
        super(lstm, self).__init__()
        # lstm
        self.add_feats = add_feats
        self.conv_i = nn.Sequential(
            nn.Conv2d(nChannels + self.add_feats, nChannels, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(nChannels + self.add_feats, nChannels, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(nChannels + self.add_feats, nChannels, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(nChannels + self.add_feats, nChannels, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, h, c):
        # x[0] extract from rain image, x[1] extract from rain estimate image, x[2] extract from segment image
        out = torch.cat((x, h.cuda()), 1)
        i = self.conv_i(out)
        f = self.conv_f(out)
        g = self.conv_g(out)
        o = self.conv_o(out)
        c_t = f * c.cuda() + i * g
        h_t = o * torch.tanh(c_t)
        out = h_t
        return out, h_t, c_t

class _asspmodule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_asspmodule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class aspp(nn.Module):
    def __init__(self, inc=64, feat=8, outc=16, dilations = [1, 6, 12, 18], BatchNorm=SynchronizedBatchNorm2d):
        super(aspp, self).__init__()

        self.aspp1 = _asspmodule(inc, feat, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _asspmodule(inc, feat, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _asspmodule(inc, feat, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _asspmodule(inc, feat, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inc, feat, 1, stride=1, bias=False),
                                             BatchNorm(feat),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(feat * 5, outc, 1, bias=False)
        self.bn1 = BatchNorm(outc)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

