import numpy as np
import spiepy
from pathlib import Path
import cv2

def line_flatten(image):
    image = np.array(image)
    output = np.zeros(image.shape)
    for i, row in enumerate(image):
        mean = np.mean(row)
        output[i] = row - mean
    
    return output

def basic_flatten(image):
    im = spiepy.Im()
    im.data = image
    im, _ = spiepy.flatten_xy(im)
    im, _ = spiepy.flatten_poly_xy(im, deg=2)
    im = line_flatten(im.data)
    im = cv2.GaussianBlur(im,(3,3), cv2.BORDER_DEFAULT)
    return im

def flatten(images):
    output = []
    for image in images:
        try:
            flattened = basic_flatten(image)
            output.append(flattened)

        except Exception as error:
            print(error)
            output.append(image)
    
    return output