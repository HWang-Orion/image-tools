from typing import Dict, Tuple, Union, List
import numpy as np
import cv2 as cv
import os
from tqdm import tqdm


# def add_wm_to_img(img: np.ndarray, wm_info: Dict, wm_type: str, loc: Tuple[float, float, str] = (0.97, 0.02, "upper left"), wm_scale: Union[float, Tuple[float, float]] = 0.0125) -> np.ndarray:
#     height, width, channels = img.shape
#     # todo tuple of wm sizes
#     wm_scale = np.array(wm_scale)
#     assert 0 < wm_scale.all() <= 1 and len(wm_scale) <= 2, ValueError("Invalid watermark scale inputs!")
#     if "app" in wm_type:
#         if loc is not None:
#             assert 0 <= loc[0] <= 1 and 0 <= loc[1] <= 1, ValueError("Invalid location")
#             assert loc[2] in ("center", "upper left", "lower left")
#         else:
#             loc = (np.random.rand(), np.random.rand(), "upper left")
#         # only treat position as upper left corner for now
#         if 'apparent_wm' in wm_info.keys() and wm_info['apparent_wm'] is not None:
#             # adding apparent water mark
#             app_img = np.copy(wm_info['apparent_wm'])
#             app_img_ratio = app_img.shape[1]  / app_img.shape[0]
#             s = wm_scale[0] if len(wm_scale) == 2 else wm_scale
#             app_img_height = int(height * s)
#             app_img = cv.resize(app_img, (int(app_img_height * app_img_ratio), app_img_height))
#             roi = img[int(height * loc[0]): int(height * loc[0]) + app_img.shape[0], 
#                       int(width * loc[1]): int(width * loc[1]) + app_img.shape[1], 
#                       0:3]
#             roi = np.where(app_img[:, :, -1:] == 0, roi, app_img[:, :, 0:3])
#             img[int(height * loc[0]): int(height * loc[0]) + app_img.shape[0], 
#                 int(width * loc[1]): int(width * loc[1]) + app_img.shape[1], 
#                 0:3] = roi
#     if "dark" in wm_type and wm_info['dark_wm'] is not None:
#         loc = (np.random.rand() * 0.8 + 0.1, np.random.rand() * 0.8 + 0.1, "upper left")
#         if isinstance(wm_info['dark_wm'], np.ndarray):
#             dk_img = wm_info['dark_wm']
#             # this is an array
#             try:
#                 diag = wm_info['dark_wm_diag']
#             except KeyError:
#                 diag = int(np.linalg.norm(wm_info['dark_wm'].shape[0:2]))
#             finally:
#                 center = (dk_img.shape[0]/2, dk_img.shape[1]/2)
#                 angle = np.random.rand() * 180 - 90
#                 s = wm_scale[1] if len(wm_scale) == 2 else wm_scale
#                 rot_matrix = cv.getRotationMatrix2D(center, angle, s)
#                 rotated_dk_img = cv.warpAffine(dk_img, rot_matrix, (diag + 400, diag + 400))
#                 rotated_dk_img = crop_nonzero(rotated_dk_img)
#                 # cv.imshow("dark", rotated_dk_img)
#                 # print(rotated_dk_img[:, :, -1])
#                 roi = img[int(height * loc[0]): int(height * loc[0]) + rotated_dk_img.shape[0], 
#                       int(width * loc[1]): int(width * loc[1]) + rotated_dk_img.shape[1], 
#                       0:3]
#                 # todo change the opacity of the wm
#                 roi = np.where(rotated_dk_img[:, :, -1:] == 0, roi, rotated_dk_img[:, :, 0:3])
#                 img[int(height * loc[0]): int(height * loc[0]) + rotated_dk_img.shape[0], 
#                     int(width * loc[1]): int(width * loc[1]) + rotated_dk_img.shape[1], 
#                     0:3] = roi
#                 # print(loc[0], loc[1])
#         elif isinstance(wm_info['dark_wm'], str):
#             # this is a string
#             pass
    
#     return img


# def crop_nonzero(image: np.ndarray):
#     # borrowed from https://stackoverflow.com/a/59208291
#     y_nonzero, x_nonzero, _ = np.nonzero(image)
#     return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def read_images(path: str) -> List[np.ndarray]:
    images = []
    img_names = os.listdir(path)
    num_of_images = len(img_names)
    print(f"{len(os.listdir(path))} image(s) founded in {path}.")
    pbar = tqdm(desc="Loading images", total=num_of_images)
    for imname in img_names:
        images.append(os.path.join(path, imname))
    return images


def check_folder_size(path: str,) -> int:
    size = 0
    try:
        for ele in os.scandir(path):
            size += os.stat(ele).st_size
        return size/1e9
    except FileNotFoundError:
        raise ValueError("Invalid path given: " + path)
