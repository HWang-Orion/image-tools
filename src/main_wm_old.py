import numpy as np
import cv2 as cv
import argparse
import os
from tqdm import tqdm

from . import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_folder', type=str, default='img/', help="Image folder")
    parser.add_argument('--wm_folder', type=str, default='wm/', required=False, help="Watermark folder, if you want to use preset images.")
    parser.add_argument('--apparent_wm_filename', type=str, default='apparent_wm.png', required=False)
    parser.add_argument('--dark_wm_filename', type=str, default='dark_wm.png', required=False)
    parser.add_argument('--dark_wm_text', type=str, default='Photo by ...', required=False)
    parser.add_argument('--output_type', type=str, default='jpg', required=False)
    parser.add_argument('--output_folder', type=str, default='output', required=False)
    
    args = parser.parse_args()
    wm_info = {}
    
    wm_path = os.path.join(os.getcwd(), args.wm_folder)
    wms = os.listdir(wm_path)
    if not args.dark_wm_filename in wms:
        wm_info['dark_wm'] = args.dark_wm_text if args.dark_wm_text else None
    else:
        wm_info['dark_wm'] = cv.imread(os.path.join(wm_path, args.dark_wm_filename), cv.IMREAD_UNCHANGED)
        wm_info['dark_wm_diag'] = int(np.linalg.norm(wm_info['dark_wm'].shape[0:2]))
    if not args.apparent_wm_filename in wms:
        wm_info['apparent_wm'] = None
    else:
        wm_info['apparent_wm'] = cv.imread(os.path.join(wm_path, args.apparent_wm_filename), cv.IMREAD_UNCHANGED)
        
    
    image_path = os.path.join(os.getcwd(), args.image_folder)
    out_path = os.path.join(os.getcwd(), args.output_folder)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    
    print(f"{len(os.listdir(image_path))} image(s) founded in {image_path}.")
    for img in tqdm(os.listdir(image_path)):
        im = cv.imread(os.path.join(image_path, img))
        if args.output_type == "jpg":
            cv.imwrite(os.path.join(out_path, img), utils.add_wm_to_img(im, wm_info, "app+dark", wm_scale=(0.0125, 0.15)), [int(cv.IMWRITE_JPEG_QUALITY), 100])
    
    print(f"Job finished. Output folder: {out_path}.")
    # cv.waitKey(0)
