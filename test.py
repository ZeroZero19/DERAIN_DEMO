#!/usr/bin/env python
import argparse
import glob
from models import *
from utils.utils import *
from torchvision.utils import make_grid, save_image
from pytorch_fid.fid_score import calculate_fid_given_paths


parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str, default="testset/our_testset", help='path of data files')
parser.add_argument('--out', type=str, default='result/our_testset', help='folder to output')
parser.add_argument('--checkpoints', type=str, default='checkpoints/M2GAN_our_testset', help='model checkpoints')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.set_defaults(feature=True)

opt = parser.parse_args()
print(opt)

# GPU
ngpu = opt.nGPU
if ngpu==1:
    gpu = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# semantic initial
semantic = load_sementic()

# load checkpoint
checkpoint = torch.load(opt.checkpoints)
model_type = checkpoint['type']
inxseg_chs = (150 if model_type != 'No-Disc' else 0)
t = checkpoint['num_stage']
GNet = dict()
for t_ in range(t):
    if t_ <= 1:
        if model_type == 'M2GAN-raindrop':
            GNet[t_] = RDUNet_LSTM(sn=False, use_lstm=False).to(gpu)
        else:
            GNet[t_] = RDUNet_LSTM_dilations(sn=False, use_lstm=False, inxseg_chs=0, t=0, dilations=False).to(gpu)
    else:
        if model_type == 'M2GAN-raindrop':
            GNet[t_] =  RDUNet_LSTM_dilations(sn=False, use_lstm=True, nFeats=32, inxseg_chs=inxseg_chs, t=t_, dilations=True).to(gpu)
        else:
            GNet[t_] = RDUNet_LSTM_dilations(sn=False, use_lstm=True, inxseg_chs=inxseg_chs, t=t_, dilations=True).to(gpu)
    GNet[t_] = nn.DataParallel(GNet[t_], list(range(ngpu)))
    GNet[t_].load_state_dict(checkpoint['stage%d' % t_])
    GNet[t_].eval()

# load raindata path
rainpaths = glob.glob(os.path.join(opt.data, 'data', '*_rain.png'))
alphanum = lambda key: int(os.path.basename(key).split('_')[0])
rainpaths.sort(key=alphanum)
print('num test imgs = %d' % (len(rainpaths)))

# Create out folder
os.makedirs(os.path.join(opt.out, model_type), exist_ok=True)

# Derain
derain_psnrs, derain_ssims = [], []

with torch.no_grad():
    for i, path in enumerate(rainpaths):
        img_name = os.path.basename(path).split('_')[0]
        img_rain = cv2.imread(path)
        img_rain = align_to_four(img_rain)
        img_rain = create_tensor(img_rain, gpu)
        img_gt = cv2.imread(path.replace('/data/','/gt/').replace('_rain.png','_clean.png'))
        img_gt = align_to_four(img_gt)
        img_gt = create_tensor(img_gt, gpu)

        xt = []
        for t_ in range(t):
            if t_ <= 1:
                out = GNet[t_](img_rain)
            else:
                if inxseg_chs != 0:
                    seg_est = semantic.cal_seg_est(out, gpu)
                    out = GNet[t_](img_rain, xt, seg_est)
                else:
                    out = GNet[t_](img_rain, xt)
            xt.append(out)

        derain_psnr = cal_psnr(tensor1=out, tensor2=img_gt)
        derain_ssim = cal_ssim(tensor1=out, tensor2=img_gt)
        derain_psnrs.append(derain_psnr)
        derain_ssims.append(derain_ssim)

        print('img:%s, type:%s,  psnr:%.2f,  ssim:%.4f' % (path, model_type, derain_psnr, derain_ssim))

        logimg = make_grid(out.data.clamp(0., 1.), nrow=8, normalize=False, scale_each=False)
        save_image(logimg, os.path.join(opt.out,model_type, '%s_%s.png' % (img_name, model_type)))

    print('MEAN, data:%s, type:%s,  psnr:%.2f,  ssim:%.4f' % (opt.data, model_type, np.array(derain_psnrs).mean(), np.array(derain_ssims).mean()))

fid = calculate_fid_given_paths(
        paths=[os.path.join(opt.data, 'gt'),os.path.join(opt.out,model_type)],
        batch_size=1,
        cuda='True',
        dims=2048,
    )
print('FID of %s: %.4f' % (type, fid))
