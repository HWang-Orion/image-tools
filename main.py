from typing import Dict, Tuple
import numpy as np
import cv2 as cv
import argparse
import os
from tqdm import tqdm

from typing import Tuple, Dict, Union


def add_wm_to_img(img: np.ndarray, wm_info: Dict, wm_type: str, loc: Tuple[float, float, str] = (0.97, 0.02, "upper left"), wm_scale: Union[float, Tuple[float, float]] = 0.0125) -> np.ndarray:
    height, width, channels = img.shape
    # todo tuple of wm sizes
    wm_scale = np.array(wm_scale)
    assert 0 < wm_scale.all() <= 1 and len(wm_scale) <= 2, ValueError("Invalid watermark scale inputs!")
    if "app" in wm_type:
        if loc is not None:
            assert 0 <= loc[0] <= 1 and 0 <= loc[1] <= 1, ValueError("Invalid location")
            assert loc[2] in ("center", "upper left", "lower left")
        else:
            loc = (np.random.rand(), np.random.rand(), "upper left")
        # only treat position as upper left corner for now
        if 'apparent_wm' in wm_info.keys() and wm_info['apparent_wm'] is not None:
            # adding apparent water mark
            app_img = np.copy(wm_info['apparent_wm'])
            app_img_ratio = app_img.shape[1]  / app_img.shape[0]
            s = wm_scale[0] if len(wm_scale) == 2 else wm_scale
            app_img_height = int(height * s)
            app_img = cv.resize(app_img, (int(app_img_height * app_img_ratio), app_img_height))
            roi = img[int(height * loc[0]): int(height * loc[0]) + app_img.shape[0], 
                      int(width * loc[1]): int(width * loc[1]) + app_img.shape[1], 
                      0:3]
            roi = np.where(app_img[:, :, -1:] == 0, roi, app_img[:, :, 0:3])
            img[int(height * loc[0]): int(height * loc[0]) + app_img.shape[0], 
                int(width * loc[1]): int(width * loc[1]) + app_img.shape[1], 
                0:3] = roi
    if "dark" in wm_type and wm_info['dark_wm'] is not None:
        loc = (np.random.rand() * 0.8 + 0.1, np.random.rand() * 0.8 + 0.1, "upper left")
        if isinstance(wm_info['dark_wm'], np.ndarray):
            dk_img = wm_info['dark_wm']
            # this is an array
            try:
                diag = wm_info['dark_wm_diag']
            except KeyError:
                diag = int(np.linalg.norm(wm_info['dark_wm'].shape[0:2]))
            finally:
                center = (dk_img.shape[0]/2, dk_img.shape[1]/2)
                angle = np.random.rand() * 180 - 90
                s = wm_scale[1] if len(wm_scale) == 2 else wm_scale
                rot_matrix = cv.getRotationMatrix2D(center, angle, s)
                rotated_dk_img = cv.warpAffine(dk_img, rot_matrix, (diag + 400, diag + 400))
                rotated_dk_img = crop_nonzero(rotated_dk_img)
                # cv.imshow("dark", rotated_dk_img)
                # print(rotated_dk_img[:, :, -1])
                roi = img[int(height * loc[0]): int(height * loc[0]) + rotated_dk_img.shape[0], 
                      int(width * loc[1]): int(width * loc[1]) + rotated_dk_img.shape[1], 
                      0:3]
                # todo change the opacity of the wm
                roi = np.where(rotated_dk_img[:, :, -1:] == 0, roi, rotated_dk_img[:, :, 0:3])
                img[int(height * loc[0]): int(height * loc[0]) + rotated_dk_img.shape[0], 
                    int(width * loc[1]): int(width * loc[1]) + rotated_dk_img.shape[1], 
                    0:3] = roi
                # print(loc[0], loc[1])
        elif isinstance(wm_info['dark_wm'], str):
            # this is a string
            pass
    
    return img


def crop_nonzero(image: np.ndarray):
    # borrowed from https://stackoverflow.com/a/59208291
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


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
            cv.imwrite(os.path.join(out_path, img), add_wm_to_img(im, wm_info, "app+dark", wm_scale=(0.0125, 0.15)), [int(cv.IMWRITE_JPEG_QUALITY), 100])
    
    print(f"Job finished. Output folder: {out_path}.")
    # cv.waitKey(0)
