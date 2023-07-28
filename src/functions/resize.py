import cv2 as cv
import numpy as np
from typing import Tuple


def resize(img: np.ndarray, resol: Tuple[int, int],
           change_aspect_ratio: bool = False, center_crop: bool = False) -> np.ndarray:
    target_asp_ratio = resol[1] / resol[0]  # resol example (2160, 3840)
    height, width, channels = img.shape
    sr_flag = False
    if width < resol[1] or height < resol[0]:
        sr_flag = True
        print("Image might be enlarged. Currently no specific super-resolution algorithms are implemented." +
              " Enlarging might not yield desired output.")
    img_asp_ratio = width / height
    if not center_crop:
        if abs(img_asp_ratio - target_asp_ratio) < 1e-3 or change_aspect_ratio:
            resol_ = resol if img_asp_ratio <= 1 else (resol[1], resol[0])
            return cv.resize(img, resol_, interpolation=cv.INTER_CUBIC)
        else:
            # match longer dimension
            if img_asp_ratio >= 1:  # width > height, match width
                return cv.resize(img, (int(resol[1] * img_asp_ratio), resol[1]), interpolation=cv.INTER_CUBIC)
            else:  # height > width
                return cv.resize(img, (int(resol[0]), int(resol[0] / img_asp_ratio)), interpolation=cv.INTER_CUBIC)
    else:
        # crop and change aspect ratio
        if not sr_flag:
            # no super resolution -> directly crop
            # img_ = np.zeros((resol[0], resol[1], channels))
            center_y, center_x = height // 2, width // 2
            img_ = img[center_y - int(resol[0] // 2): center_y + int(resol[0] / 2),
                       center_x - int(resol[1] // 2): center_x + int(resol[1] / 2), 0:channels]
            return img_
        else:
            raise NotImplementedError("Super resolution with center cropped image is not implemented.")
