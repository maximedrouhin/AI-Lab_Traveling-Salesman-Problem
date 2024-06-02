import cv2
import numpy as np

def neighbour(loc1, loc2):
    if abs(loc1[0]-loc2[0]) <= 2 and abs(loc1[1]-loc2[1]) <= 2:
        return True
    return False

def part_of_circle(loc, circle):
    for c in circle:
        if neighbour(loc, c):
            return True
    return False

def same_circle(circle1, circle2):
    for c1 in circle1:
        if part_of_circle(c1, circle2):
            return True
    return False

def get_yellow_circles_my1(image):

    # Define lower and upper bounds for yellow color in HSV
    lower_yellow = 150


    # Threshold the image to obtain a binary mask of yellow regions
    yellow_mask = np.where((image[:, :, 0] > lower_yellow), 255, 0)

    yellow_circles = []

    positions = np.argwhere(yellow_mask)

    for pos in positions:
        tmp = False
        for circle in yellow_circles:
            if part_of_circle(pos, circle):
                circle.append(pos)
                tmp = True
                break
        if not tmp:
            yellow_circles.append([pos])

    for circle1 in yellow_circles:
        for circle2 in yellow_circles:
            if np.array_equal(circle1[0], circle2[0]):
                continue
            if same_circle(circle1, circle2):
                circle1 += circle2
                yellow_circles.remove(circle2)

    yellow_circles_center = []
    for circle in yellow_circles:
        x_min = np.min(circle, axis=0)[1]
        x_max = np.max(circle, axis=0)[1]
        y_min = np.min(circle, axis=0)[0]
        y_max = np.max(circle, axis=0)[0]
        yellow_circles_center.append([(x_max-x_min)/2+x_min, (y_max-y_min)/2+y_min])

    return yellow_circles_center



def get_yellow_circles_cv2(image):
    
        # Define lower and upper bounds for yellow color in HSV
        lower_yellow = np.array([50, 0, 0])
        upper_yellow = np.array([255, 255, 255])

    
        # Threshold the image to obtain a binary mask of yellow regions
        yellow_mask = cv2.inRange(image, lower_yellow, upper_yellow)
    
        # Find contours in the binary mask
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
        # Draw the contours on the original image
        yellow_circles = []
        for contour in contours:
            # Compute the center of the contour
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            yellow_circles.append([x, y])
    
        return np.array(yellow_circles)