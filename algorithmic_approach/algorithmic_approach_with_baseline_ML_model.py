import csv
import numpy as np
import pandas as pd
import os
import matplotlib.image as mpimg
from functools import partial
import time

import functions_circle_detection as cdf
import functions_line_detection as ld
import functions_pathfinding as pf

import keras
from keras_preprocessing import image


start_time = time.time()
print('start time:', time.strftime('%H:%M:%S', time.localtime(start_time)))

#folder structure
data_folder = 'tsp-cv/'

train_data = pd.read_csv(os.path.join(data_folder,'train.csv'))


export_folder = 'results/'
export_name = 'results_over_night.csv'

imagenumbers = range(100, 200)

#functions for circle detection, starting position detection, connection matrix and the pathfinding functions
circle_detection_function = cdf.get_yellow_circles_cv2
starting_position_function = cdf.get_green_circle
connection_matrix_function = pf.get_connection_matrix
#connection_matrix_function = partial(pf.get_connection_matrix, score_function=lambda x: 1 - x, line_detection_function=ldf.get_ratio_blue_pixels)
pathfinder_functions = {
    'simple': pf.get_simple_best_path,
    'tsp': pf.tsp_solver1,
    'valid': partial(pf.get_valid_path_backcheck_multiple, start=0.01, stop=0.3, step=0.02 ,zero_matr=False, maxtime=30),
    'valid_zero': partial(pf.get_valid_path_backcheck_multiple, start=0.01, stop=0.2, step=0.02 ,zero_matr=True, maxtime=30)

}


#ai model
model = keras.models.load_model("model.keras")
WIDTH = 256
HEIGHT = 256



#open csv file and write header
with open(os.path.join(export_folder, export_name), mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ['imagenumber', 'true_length', 'matrix_calc_time']
    header = header + [f'{name}_length' for name in pathfinder_functions.keys()]
    header = header + [f'{name}_path' for name in pathfinder_functions.keys()]
    header = header + [f'{name}_time' for name in pathfinder_functions.keys()]
    header = header + ['AI_prediction', 'AI_time']

    writer.writerow(header)

#for each image, calculate the path and write it to the csv file
for i,row in train_data.iterrows():

    im_nr = row['id']
    true_length = row['distance']
    filename = row['filename']

    print(im_nr)

    #load image and get true length
    img = mpimg.imread(os.path.join(data_folder, filename))

    #get positions and connection_matrix
    matr_time_start = time.time()
    positions = list(circle_detection_function(img))
    start_pos = starting_position_function(img)
    positions.insert(0, start_pos)
    connection_matrix = connection_matrix_function(img, positions)
    matr_time = time.time() - matr_time_start

    #calculate path and length for each function
    tmp_save = {}
    for name, function in pathfinder_functions.items():
        path_start = time.time()
        path = function(connection_matrix)
        path_time = time.time() - path_start
        if path is None:
            tmp_save[name] = [None, None, path_time]
            continue
        sorted_positions = [positions[i] for i in path]
        length = 0
        for i in range(1, len(sorted_positions)):
            length += np.linalg.norm(np.array(sorted_positions[i]) - np.array(sorted_positions[i-1]))
        tmp_save[name] = [length, sorted_positions, path_time]

    #AI prediction
    # Load the image
    img = image.load_img(os.path.join(data_folder, filename), target_size=(HEIGHT, WIDTH))

    img = image.img_to_array(img)/255
    img = np.array([img])

    # Make a prediction with the model
    ai_time_start = time.time()
    prediction = model.predict((img))
    ai_time = time.time() - ai_time_start

    next_row = [im_nr, true_length, matr_time]
    for name in pathfinder_functions.keys():
        next_row.append(tmp_save[name][0])
    for name in pathfinder_functions.keys():
        next_row.append(tmp_save[name][1])
    for name in pathfinder_functions.keys():
        next_row.append(tmp_save[name][2])

    next_row.append(prediction[0][0])
    next_row.append(ai_time)

    with open(os.path.join(export_folder, export_name), mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(next_row)


print('end time:', time.strftime('%H:%M:%S', time.localtime(time.time())))
print('total time:', time.time() - start_time)
print(time.strftime('%H:%M:%S', time.time() - start_time))