import cv2
import numpy as np

def get_ratio_blue_pixels(image,start,end):
    black_picture = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    image_cp = image[:,:,2].copy()
    image_cp = image_cp.astype(np.int16)
    image_cp -= 50 # Threshold for blue color
    image_cp = np.where(image_cp > 0, image_cp, 0)
    cv2.line(black_picture, start, end, 100, 2)
    image_cp = np.where(black_picture > 0, image_cp, 0)
    #print(np.count_nonzero(image_cp)/np.count_nonzero(black_picture))
    return np.count_nonzero(image_cp)/np.count_nonzero(black_picture)

def get_next_pos(image, start, remaining):
    next_pos = 0
    max_ratio_blue = get_ratio_blue_pixels(image, start, remaining[0])
    for i in range(len(remaining)):
        blue_ratio = get_ratio_blue_pixels(image, start, remaining[i])
        if blue_ratio > max_ratio_blue:
            max_ratio_blue = blue_ratio
            next_pos = i
    return next_pos