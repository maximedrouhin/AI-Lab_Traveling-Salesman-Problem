import cv2
import numpy as np

def get_ratio_blue_pixels(image,start,end):
    start = np.array(start)
    end = np.array(end)
    vector = end - start
    #get a normalized vector
    norm_vector = vector / np.linalg.norm(vector)
    amount = np.ceil(np.linalg.norm(vector)).astype(int)
    nonzero=0
    for i in [-1,0,1]:
        xx = np.linspace(start[0]+i*norm_vector[1], end[0]+i*norm_vector[1], amount).astype(int)
        yy = np.linspace(start[1]-i*norm_vector[0], end[1]-i*norm_vector[0], amount).astype(int)
        nonzero += np.count_nonzero(image[yy, xx, 2]>50)
    return nonzero / (3*amount)

def get_fast_ratio_blue_pixels(image,start,end):
    start = np.array(start)
    end = np.array(end)
    amount = np.ceil(np.linalg.norm(end-start)).astype(int)
    xx = np.linspace(start[0], end[0], amount).astype(int)
    yy = np.linspace(start[1], end[1], amount).astype(int)
    blue = image[yy, xx, 2]
    return np.count_nonzero(blue > 50) / len(blue)


def get_weighted_ratio_blue_pixels(image, start, end):
    # Convert the image from RGB to HSL
    hsl_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # Create a mask for blue pixels based on your findings
    hue = hsl_image[:,:,0] * 2  # OpenCV hue is 0-180; multiply by 2 to get 0-360
    lightness = hsl_image[:,:,1]
    blue_mask = (hue > 170) & (hue < 225) & (lightness > 20)

    # Create a blank image to draw the line
    line_image = np.zeros_like(blue_mask, dtype=np.uint8)

    # Draw the line on the blank image
    cv2.line(line_image, start, end, 1, 2)

    # Find the overlap of the blue pixels and the line
    overlap_mask = blue_mask & (line_image > 0)

    # Calculate the score based on the green component of the RGB image
    green_component = image[:,:,1]
    weighted_score = np.sum(green_component[overlap_mask])

    # Normalize the score by the length of the line
    line_length = np.count_nonzero(line_image)

    if line_length == 0:
        return 0

    return weighted_score / line_length

def get_weighted_ratio_blue_pixels_with_black_pixel_penalty(image, start, end, k=200, x0=0.02):
    # Convert the image from RGB to HSL
    hsl_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # Create a mask for blue pixels based on your findings
    hue = hsl_image[:,:,0] * 2  # OpenCV hue is 0-180; multiply by 2 to get 0-360
    lightness = hsl_image[:,:,1]
    blue_mask = (hue > 170) & (hue < 225) & (lightness > 20)

    # Create a blank image to draw the line
    line_image = np.zeros_like(blue_mask, dtype=np.uint8)

    # Draw the line on the blank image
    cv2.line(line_image, start, end, 1, 2)

    # Find the overlap of the blue pixels and the line
    overlap_mask = blue_mask & (line_image > 0)

    # Calculate the score based on the green component of the RGB image
    green_component = image[:,:,1]
    weighted_score = np.sum(green_component[overlap_mask])

    # Normalize the score by the length of the line
    line_length = np.count_nonzero(line_image)

    if line_length == 0:
        return 0

    blue_ratio = weighted_score / line_length

    # Calculate the proportion of black pixels along the line (lightness < 7)
    black_mask = (lightness < 7) & (line_image > 0)
    black_ratio = np.count_nonzero(black_mask) / line_length

    # Apply the logistic penalty function
    penalty = 1 / (1 + np.exp(k * (black_ratio - x0)))

    # Combine the blue ratio with the penalty
    transformed_score = blue_ratio * penalty

    return transformed_score

