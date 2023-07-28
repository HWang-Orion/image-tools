import cv2 as cv
import numpy as np
from typing import Union, List, Dict, Tuple
from tqdm import tqdm


pos_x_dict = {
    "upper": 0.05,
    "mid": 0.5,
    "lower": 0.95,
}
pos_y_dict = {
    "left": 0.05, 
    "mid": 0.5,
    "right": 0.95,
}


"""
def add_wm_to_img(img: np.ndarray, wm_type: str, wm_img: np.ndarray, loc: Tuple[float, float, str] = (0.97, 0.02, "upper left"), wm_scale: Union[float, Tuple[float, float]] = 0.0125) -> np.ndarray:
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
        if wm_img:
            # adding apparent water mark
            app_img = np.copy(wm_img)
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
    if "dark" in wm_type and wm_img is not None:
        loc = (np.random.rand() * 0.8 + 0.1, np.random.rand() * 0.8 + 0.1, "upper left")
        if isinstance(wm_img, np.ndarray):
            dk_img = wm_img
            # this is an array
            try:
                diag = wm_img
            except KeyError:
                diag = int(np.linalg.norm(wm_img.shape[0:2]))
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
"""


def add_wm_to_img(img: np.ndarray, wm: np.ndarray, wm_scale: float, wm_ori: float, pos: Tuple[float, float]) -> np.ndarray:
    # scale: relative to the image
    assert len(img.shape) == 3
    height, width, channels = img.shape
    im_diag = np.sqrt(height * height + width * width)
    
    if wm.shape[2] == 1:
        wm = np.tile(wm, (1, 1, 3))
    if wm.shape[2] == 2:
        raise ValueError(f"Wrong watermark image given! Expected to get one or three channels but got {wm.shape[2]} instead")
    # wm = cv.resize(wm, (int(wm.shape[0] * wm_scale), int(wm.shape[1] * wm_scale)))
    
    # watermark rotation and scaling
    center = (wm.shape[0]/2, wm.shape[1]/2)
    wm_diag = np.sqrt(wm.shape[0] ** 2 + wm.shape[1] ** 2)
    wm_scale = wm_scale * im_diag / wm_diag  # change it to the scale w.r.t. the watermark size
    rot_matrix = cv.getRotationMatrix2D(center, wm_ori, wm_scale)  # wm_scale: absolute
    wm = cv.warpAffine(wm, rot_matrix, (int(wm_diag) + 400, int(wm_diag) + 400))
    wm = crop_nonzero(wm)
    
    wm_center = [height * pos[0], width * pos[1]]
    pos_judgements = [wm_center[0] < wm.shape[0]/2, wm_center[0] > height - wm.shape[0]/2,
                      wm_center[1] < wm.shape[1]/2, wm_center[1] > width - wm.shape[1]/2]
    if True in pos_judgements:
        # watermark get out of boundary

        # the case when the watermark could not fit the entire image
        if (pos_judgements[0] and pos_judgements[1]) or (pos_judgements[2] and pos_judgements[3]):
            raise ValueError("Improper position given for the watermark")

        # four cases when the watermark gets out of boundary -> shift back
        if pos_judgements[0]:
            wm_center[0] += wm_center[0] - wm.shape[0] / 2
        if pos_judgements[1]:
            wm_center[0] -= wm_center[0] - wm.shape[0] / 2
        if pos_judgements[2]:
            wm_center[1] += wm_center[1] - wm.shape[1] / 2
        if pos_judgements[3]:
            wm_center[1] -= wm_center[1] - wm.shape[1] / 2

    # get the region of interest and add watermark; replace the corresponding part in the image with the RoI again
    roi = img[int(height * pos[0]): int(height * pos[0]) + wm.shape[0],
              int(width * pos[1]): int(width * pos[1]) + wm.shape[1], 0:3]
    # todo adaptative watermark
    roi = np.where(wm[:, :, -1:] == 0, roi, wm[:, :, 0:3])
    img[int(height * pos[0]): int(height * pos[0]) + wm.shape[0],
        int(width * pos[1]): int(width * pos[1]) + wm.shape[1], 0:3] = roi

    return img
    

def crop_nonzero(image: np.ndarray):
    # borrowed from https://stackoverflow.com/a/59208291
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def randrange(a: float, b: float):
    return np.random.rand() * (b - a) + a


def add_watermark(wm: np.ndarray, images: Union[np.ndarray, List[np.ndarray]], **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
    if isinstance(images, np.ndarray):
        images = [images]
    
    wm_pos = kwargs["wm_pos"] if "wm_pos" in kwargs.keys() else "random"
    
    if wm_pos != "random":
        if isinstance(wm_pos, tuple):
            assert len(wm_pos) == 2 and 0 <= wm_pos[0] <= 1 and 0 <= wm_pos[1] <= 1, ValueError(f"Invalid position! Expected to be a tuple with two elements in range [0, 1], got {wm_pos} instead.")
        elif isinstance(wm_pos, str):
            pos_x, pos_y = tuple(wm_pos.split())
            assert pos_x in pos_x_dict.keys() and pos_y in pos_y_dict.keys()
            wm_pos = (pos_x_dict[pos_x], pos_y_dict[pos_y])
        else:
            raise ValueError("Invalid position!")
    
    wm_scale = kwargs["wm_scale"] if "wm_scale" in kwargs.keys() else 0.0625
    
    wm_ori = kwargs["wm_ori"] if "wm_ori" in kwargs.keys() else "random"
    if wm_ori != "random":
        wm_ori = float(wm_ori) % 360

    images_ = []

    for img in tqdm(images):
        if wm_pos == "random":
            wm_pos = (randrange(0.15, 0.85), randrange(0.15, 0.85))
        if wm_ori == "random":
            wm_ori = np.random.random() * 180 - 90
        images_.append(add_wm_to_img(img, wm, wm_scale, wm_ori, wm_pos))

    return images_ if len(images_) > 1 or isinstance(images, list) else images_[0]
