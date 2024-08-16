import os
import math
import cv2
import torch

import numpy as np
import pandas as pd
import torchvision.transforms.functional as F



from typing import Union, List, Tuple
from PIL.Image import Image
from torch import Tensor






root_dir = "../data/gehler/"
BOARD_FILL_COLOR = 1e-5





















################## Pre-processing for dataset ##################

def get_mcc_coord(fn: str) -> np.ndarray:
    """
    Func : computes the relative MCC coordinates for the given image
    Args: file name ex) 'IMG_0369.png'
    Return: relative MCC coordinates (4 points)
    """
    lines = open(root_dir + 'coordinates/' + fn.split('.')[0] + '_macbeth.txt', 'r').readlines()
    width, height = map(float, lines[0].split())
    scale_x, scale_y = 1 / width, 1 / height
    polygon = []
    for line in [lines[1], lines[2], lines[4], lines[3]]:
        line = line.strip().split()
        x, y = (scale_x * float(line[0])), (scale_y * float(line[1]))
        polygon.append((x, y))
    return np.array(polygon, dtype='float32')


def load_image(fn: str) -> np.ndarray:
    """
    Func : load image with color checker
    Args: file name ex) 'IMG_0369.png'
    Return: array of RAW image
    """
    file_path = "../data/gehler/" + '/images/' + fn
    raw = np.array(cv2.imread(file_path, cv2.IMREAD_UNCHANGED), dtype='float32') # RAW image
    # Handle pictures taken with Canon 5d Mark III
    black_point = 129 if fn.startswith('IMG') else 1
    # Keep only the pixels such that raw - black_point > 0
    raw = np.maximum(raw - black_point, [0, 0, 0])
    return raw


def load_image_without_mcc(fn : str, mcc_coord: np.ndarray) -> np.ndarray:
    """
    Load image without color checker (masking with a black polygon)
    @Param fn: file name ex) 'IMG_0369.png'
    @Param mcc_coord: mask coordinates
    @Return: array of RAW image with mask
    """
    raw = load_image(fn)
    # Clip values between 0 and 1
    img = (np.clip(raw / raw.max(), 0, 1) * 65535.0).astype(np.uint16)
    # Get vertices for polygon mask
    polygon = mcc_coord * np.array([img.shape[1], img.shape[0]])
    polygon = polygon.astype(np.int32)
    # Fill the polygon to image
    cv2.fillPoly(img, pts=[polygon], color=(BOARD_FILL_COLOR,) * 3)
    return img


def normalize(img: np.ndarray) -> np.ndarray:
    max_int = 65535.0
    return np.clip(img, 0.0, max_int) * (1.0 / max_int)


# opencv - use bgr
def bgr_to_rgb(x: np.ndarray) -> np.ndarray:
    return x[:, :, ::-1] # slice: start, end, step


def rgb_to_bgr(x: np.ndarray) -> np.ndarray:
    return x[:, :, ::-1] # slice: start, end, step


def linear_to_nonlinear(img: Union[np.array, Image, Tensor]) -> Union[np.array, Image, Tensor]:
    if isinstance(img, np.ndarray):
        return np.power(img, (1.0 / 2.2))
    if isinstance(img, Tensor):
        return torch.pow(img, 1.0 / 2.2)
    return F.to_pil_image(torch.pow(F.to_tensor(img), 1.0 / 2.2).squeeze(), mode="RGB")


def correct(img: Image, illuminant: Tensor) -> Image:
    """
    Corrects the color of the illuminant of a linear image based on an estimated (linear) illuminant
    @param img: a linear image
    @param illuminant: a linear illuminant
    @return: a non-linear color-corrected version of the input image
    """
    img = F.to_tensor(img).to(DEVICE)

    # Correct the image
    correction = illuminant.unsqueeze(2).unsqueeze(3) * torch.sqrt(Tensor([3])).to(DEVICE)
    corrected_img = torch.div(img, correction + 1e-10)

    # Normalize the image
    max_img = torch.max(torch.max(torch.max(corrected_img, dim=1)[0], dim=1)[0], dim=1)[0] + 1e-10
    max_img = max_img.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    normalized_img = torch.div(corrected_img, max_img)

    return F.to_pil_image(linear_to_nonlinear(normalized_img).squeeze(), mode="RGB")


def hwc_to_chw(x: np.ndarray) -> np.ndarray:
    """
    Converts an image from height x width x channels to channels x height x width 
    """
    return x.transpose(2, 0, 1)




