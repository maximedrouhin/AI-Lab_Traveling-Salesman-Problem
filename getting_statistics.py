import csv
import numpy as np
import os
import matplotlib.image as mpimg
from functools import partial
import circle_detection_functions as cdf
import line_detection_functions as ldf
import pathfinder_functions as pf

import time

start_time = time.time()
print('start time:', time.strftime('%H:%M:%S', time.localtime(start_time)))

data_folder = 'tsp-cv/'

export_folder = 'results/'
export_name = 'results3.csv'

imagenumbers = range(100, 200)


circle_detection_function = cdf.get_yellow_circles_cv2
starting_position_function = cdf.get_green_circle
connection_matrix_function = pf.get_connection_matrix
#connection_matrix_function = partial(pf.get_connection_matrix, score_function=lambda x: 1 - x, line_detection_function=ldf.get_ratio_blue_pixels)
functions = {
    'simple': pf.get_simple_best_path,
    'tsp': pf.tsp_solver1,
    #'valid': partial(pf.get_valid_path_backcheck, max_val=0.1 ,zero_matr=True)
    'valid': partial(pf.get_valid_path_backcheck_multiple, start=0.01, stop=0.2, step=0.01 ,zero_matr=True, maxtime=20)

}



with open(os.path.join(export_folder, export_name), mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ['imagenumber', 'true_lenght']+ [f'{name}_lenght' for name in functions.keys()] + [f'{name}_path' for name in functions.keys()]
    writer.writerow(header)

for im_nr in imagenumbers:
    print(im_nr)
    image = mpimg.imread(f'tsp-cv/{im_nr}.jpg')
    with open('tsp-cv/train.csv') as f:
        for i, line in enumerate(f):
            if i == im_nr+1:
                true_lenght = int(line.split(',')[2])
                break
    positions = list(circle_detection_function(image))
    start_pos = starting_position_function(image)
    positions.insert(0, start_pos)
    connection_matrix = connection_matrix_function(image, positions)

    tmp_save = {}
    for name, function in functions.items():
        path = function(connection_matrix)
        if path is None:
            tmp_save[name] = [None, None]
            continue
        sorted_positions = [positions[i] for i in path]
        lenght = 0
        for i in range(1, len(sorted_positions)):
            lenght += np.linalg.norm(np.array(sorted_positions[i]) - np.array(sorted_positions[i-1]))
        tmp_save[name] = [lenght, sorted_positions]

    next_row = [im_nr, true_lenght]
    for name in functions.keys():
        next_row.append(tmp_save[name][0])
    for name in functions.keys():
        next_row.append(tmp_save[name][1])

    with open(os.path.join(export_folder, export_name), mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(next_row)
        

print('end time:', time.strftime('%H:%M:%S', time.localtime(time.time())))
print('total time:', time.time() - start_time)
print(time.strftime('%H:%M:%S', time.time() - start_time))