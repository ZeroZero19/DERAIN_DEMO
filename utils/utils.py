import cv2
import numpy as np
import torch
from skimage.measure import compare_psnr, compare_ssim
from semantic.config import cfg
from semantic.models import ModelBuilder, SegmentationModule
from semantic.utils import setup_logger, colorEncode

import torchvision
from scipy.io import loadmat
import csv
import os
import torch.nn as nn


def batch_psnr(img, imclean, data_range=1., batch_ssim=False):
    r"""
    Computes the PSNR along the batch dimension (not pixel-wise)

    Args:
        img: a `torch.Tensor` containing the restored image
        imclean: a `torch.Tensor` containing the reference image
        data_range: The data range of the x image (distance between
            minimum and maximum possible values). By default, this is estimated
            from the image data-type.
    """
    img_cpu = img.data.cpu().numpy().astype(np.float32)
    imgclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    ssim = 0
    for i in range(img_cpu.shape[0]):
        # psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], data_range=data_range)
        im1 = np.array(img_cpu[i, :, :, :].transpose((1, 2, 0)) * 255.0, dtype='uint8')
        im2 = np.array(imgclean[i, :, :, :].transpose((1, 2, 0)) * 255.0, dtype='uint8')
        im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
        im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
        psnr += compare_psnr(im1_y, im2_y)
        if batch_ssim:
            # ssim += compare_ssim(imgclean[i, :, :, :].transpose(1, 2, 0),
            #                      img_cpu[i, :, :, :].transpose(1, 2, 0),
            #                      data_range=data_range,
            #                      multichannel=True)
            ssim += compare_ssim(im1_y, im2_y)

    if batch_ssim:
        return psnr / img_cpu.shape[0], ssim / img_cpu.shape[0]
    else:
        return psnr/img_cpu.shape[0]

def batch_ssim(img, imclean, data_range):
    r"""
    Computes the SSIM along the batch dimension (not pixel-wise)

    Args:
        img: a `torch.Tensor` containing the restored image
        imclean: a `torch.Tensor` containing the reference image
        data_range: The data range of the x image (distance between
            minimum and maximum possible values). By default, this is estimated
            from the image data-type.
    """
    img_cpu = img.data.cpu().numpy().astype(np.float32)
    imgclean = imclean.data.cpu().numpy().astype(np.float32)
    ssim = 0
    for i in range(img_cpu.shape[0]):
        ssim += compare_ssim(imgclean[i, :, :, :].transpose(1, 2, 0),
                             img_cpu[i, :, :, :].transpose(1, 2, 0),
                             data_range=data_range,
                             multichannel=True)
    return ssim/img_cpu.shape[0]

def create_tensor(img,gpu):
    img = np.array(img[:, :, ::-1] / 255.0).astype('float32')
    img = torch.Tensor(np.array(img).transpose(2, 0, 1).astype('float32')).unsqueeze(0)
    img = img.to(gpu)
    return img


def cal_psnr(tensor1, tensor2):
    return batch_psnr(tensor1.clamp(0., 1.), tensor2.clamp(0., 1.), 1.)

def cal_ssim(tensor1, tensor2):
    return batch_ssim(tensor1.clamp(0., 1.), tensor2.clamp(0., 1.), 1.)

def align_to_four(img):
    # print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    # align to four
    a_row = int(img.shape[0] / 4) * 4
    a_col = int(img.shape[1] / 4) * 4
    img = img[0:a_row, 0:a_col]
    # print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img

class load_sementic:
    def __init__(self,):
        super(load_sementic, self).__init__()
        self.trf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.colors = loadmat('/home/manh/DERAIN_FULL/semantic/data/color150.mat')['colors']
        names = {}
        with open('/home/manh/DERAIN_FULL/semantic/data/object150_info.csv') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                names[int(row[0])] = row[5].split(";")[0]

        cfg_config_file = '/home/manh/DERAIN_FULL/semantic/config/ade20k-resnet50dilated-ppm_deepsup.yaml'
        cfg.merge_from_file(cfg_config_file)
        cfg.DIR = '/home/manh/DERAIN_FULL/semantic/ade20k-resnet50dilated-ppm_deepsup'
        # cfg.freeze()

        logger = setup_logger(distributed_rank=0)  # TODO
        logger.info("Loaded configuration file {}".format(cfg_config_file))
        logger.info("Running with config:\n{}".format(cfg))

        cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
        cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

        # absolute paths of model weights
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

        assert os.path.exists(cfg.MODEL.weights_encoder) and \
               os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

        # Network Builders
        net_encoder = ModelBuilder.build_encoder(
            arch=cfg.MODEL.arch_encoder,
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_encoder)
        net_decoder = ModelBuilder.build_decoder(
            arch=cfg.MODEL.arch_decoder,
            fc_dim=cfg.MODEL.fc_dim,
            num_class=cfg.DATASET.num_class,
            weights=cfg.MODEL.weights_decoder,
            use_softmax=True)

        crit = nn.NLLLoss(ignore_index=-1)

        self.segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

        self.segmentation_module.cuda()

        self.segmentation_module.eval()

    def cal_seg_est(self, inseg, gpu):
        # create segment x
        with torch.no_grad():
            inseg = inseg.clone()
            in_seg_trf = []
            for i_num in range(inseg.shape[0]):
                in_seg_trf.append(self.trf(inseg[i_num, :, :, :]))
            in_seg_trf = torch.stack(in_seg_trf)
            # process data
            segSize = (in_seg_trf.shape[2],
                       in_seg_trf.shape[3])
            seg_est = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1]).to(gpu)
            # forward pass
            feed_dict = dict()
            feed_dict['img_data'] = in_seg_trf.clone()
            pred_tmp = self.segmentation_module(feed_dict, segSize=segSize)
            seg_est = seg_est + pred_tmp / len(cfg.DATASET.imgSizes)
        return seg_est

    def cal_seg_color(self, seg):
        with torch.no_grad():
            _, pred_tmp = torch.max(seg, dim=1)
            pred_tmp = pred_tmp.cpu().numpy().astype(np.uint8)
            pred_colors = []
            for _ in range(pred_tmp.shape[0]):
                pred_color = colorEncode(pred_tmp[_, :, :], self.colors)
                # plt.imshow(pred_color)
                pred_color = torch.Tensor(np.array(pred_color / 255.).transpose(2, 0, 1).astype('float32')).to(gpu)
                pred_colors.append(pred_color)
            pred_colors = torch.stack(pred_colors)
        return pred_colors