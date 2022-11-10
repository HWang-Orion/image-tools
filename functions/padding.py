import numpy as np
from typing import Sequence


def padding(img: np.ndarray, ratio: Sequence[float], border: float = 0, white: bool = True) -> np.ndarray:
    height, width, channels = img.shape
    ori_ratio = width / height
    # ratio in width : height
    ratio = ratio[0] / ratio[1]  # width / height, inverse in np representation
    # e.g. given 16:9 image expand to 21:9
    if ratio > ori_ratio:
        # expand width, new dimension depend on height
        new_dim = (int(height + 2 * border * height), int(height * ratio), 3)
    else:
        # expand height
        new_dim = (int(width / ratio), int(width + 2 * border * width), 3)
    img_n = np.ones(new_dim, dtype=np.uint8) * 255 if white else np.zeros(new_dim, dtype=np.uint8)
    img_n_center = new_dim[0] // 2, new_dim[1] // 2
    img_n[int(img_n_center[0] - np.floor(height / 2)): int(img_n_center[0] + np.ceil(height / 2)),
          int(img_n_center[1] - np.floor(width / 2)): int(img_n_center[1] + np.ceil(width / 2)), :] = img
    return img_n
