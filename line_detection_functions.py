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


def get_connection_matrix(image, positions,score_function=lambda x: 1 - x) -> np.ndarray:
    connection_matrix = np.full((len(positions), len(positions)), np.inf, dtype=np.float64)
    for i in range(len(positions)-1):
        for j in range(i+1,len(positions)):
            ratio = get_ratio_blue_pixels(image, positions[i], positions[j])
            score = score_function(ratio)
            connection_matrix[i][j] = score
            connection_matrix[j][i] = score
    return connection_matrix

def get_simple_best_path(connection_matrix: np.ndarray) -> tuple[list, float]:
    matr = connection_matrix.copy()
    best_path = [0]
    best_score = 0
    for i in range(connection_matrix.shape[0]-1):
        next_pos = np.argmin(matr[best_path[-1]])
        best_path.append(next_pos)
        best_score += connection_matrix[best_path[-2]][best_path[-1]]
        matr[:,best_path[-2]] = np.inf
        matr[best_path[-2],:] = np.inf
        
    return best_path, best_score

def return_paths(connection_matrix, max_value, current_path, current_score):
    if not current_path:  #check if current_path is empty
        return []
    if len(current_path) == connection_matrix.shape[0]:
        return current_path
    print(current_path)
    paths = []
    for i in range(len(connection_matrix)):
        if i not in current_path:
            new_score = current_score + connection_matrix[current_path[-1]][i]
            if new_score <= max_value:
                new_path = current_path.copy()
                new_path.append(i)
                paths.extend(return_paths(connection_matrix, max_value, new_path, new_score))
    return paths
    

def get_best_path(connection_matrix: np.ndarray) -> tuple[list, float]:
    simple_path, simple_score = get_simple_best_path(connection_matrix)
    simple_score = 0.5
    paths = return_paths(connection_matrix, simple_score, [0], 0)
    return paths