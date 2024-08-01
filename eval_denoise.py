import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from utils import load_data
import lpips

def PSNR(gt_arr, pre_arr):
    psnr_score = cv2.PSNR(gt_arr, pre_arr)
    return psnr_score

def SSIM(gt_arr, pre_arr):
    ssim_score = ssim(gt_arr, pre_arr)
    return ssim_score

def LPIPS(gt_arr, pre_arr):
    lpips_score_alex = lpips.LPIPS(gt_arr, pre_arr,net='alex')
    lpips_score_vgg = lpips.LPIPS(gt_arr, pre_arr,net='vgg')

    return lpips_score_alex, lpips_score_vgg

def FoutierRingCorrelation(gt_arr, pre_arr):
    frc_score = cv2.frc(gt_arr, pre_arr)
    return frc_score

def write_to_txt(psnr_score,ssim_score,lpips_score_alex, lpips_score_vgg,frc_score,save_file):
    output_file = save_file
    with open(output_file, "w") as f:
        f.write("PSNR: {}\n".format(psnr_score))
        f.write("SSIM: {}\n".format(ssim_score))
        f.write("LPIPS_alex: {}\n".format(lpips_score_alex))
        f.write("LPIPS_vgg: {}\n".format(lpips_score_vgg))
        f.write("Foutier Ring Correlation: {}\n".format(frc_score))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str,
                        default='test_examples/gt.npy',
                        help='dir for ground truth volume data')
    parser.add_argument('--pre', type=str,
                        default='test_examples/gt.npy',
                        help='dir for prediction volume data')
    parser.add_argument('--save_file', type=str,
                        default='results/evaluation_results.txt',
                        help='path to save results')
    args = parser.parse_args()
    # 1. Load data
    gt_volume = load_data(args.gt)
    pre_volume = load_data(args.pre)

    # 2. compute scores
    psnr_score = PSNR(gt_volume, pred_volume)
    ssim_score = SSIM(gt_volume, pred_volume)
    lpips_score_alex, lpips_score_vgg = LPIPS(gt_volume, pred_volume)
    frc_score = FoutierRingCorrelation(gt_volume, pred_volume)

    print("PSNR:", psnr_score)
    print("SSIM:", ssim_score)
    print("LPIPS_alex:", lpips_score_alex)
    print("LPIPS_vgg:", lpips_score_vgg)
    print("Foutier Ring Correlation:", frc_score)

    write_to_txt(psnr_score,ssim_score,lpips_score_alex, lpips_score_vgg,frc_score,args.save_file)