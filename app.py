import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import as_circle_detection_functions as cdf
import as_line_detection_functions as ldf
import as_pathfinder_functions as pf
from functools import partial
import colorsys

# Initialize the detection functions
circle_detection_function = cdf.get_yellow_circles_cv2
starting_position_function = cdf.get_green_circle
connection_matrix_function = pf.get_connection_matrix
pathfinding_functions = pf.get_simple_best_path

# Sidebar
st.sidebar.title("Image Selection")
pic_num = st.sidebar.number_input("Select Image Number", min_value=0, max_value=16018, value=1, step=1)

# Load the image
image_path = f'tsp-cv/{pic_num}.jpg'
image = mpimg.imread(image_path)

# Display the original image
st.title("True and Detected Path Comparison")
# st.subheader("Original Image")
# st.image(image, caption='Original Image')

# Get the true length from the CSV
with open('tsp-cv/train.csv') as f:
    for i, line in enumerate(f):
        if i == pic_num + 1:
            true_length = int(line.split(',')[2])
            break

# Get the detected positions and connection matrix
positions = list(circle_detection_function(image))
start_pos = starting_position_function(image)
positions.insert(0, start_pos)
connection_matrix = connection_matrix_function(image, positions)

#calculate path and length for each function
path = pathfinding_functions(connection_matrix)

sorted_positions = [positions[i] for i in path]

# Helper function to generate a hue gradient
def get_hue_gradient(start_hue, end_hue, steps):
    hues = np.linspace(start_hue, end_hue, steps)
    return [colorsys.hls_to_rgb(h / 360, 0.5, 1.0) for h in hues]

# Generate the hue gradient colors
colors = get_hue_gradient(0, 308, len(sorted_positions) - 1)

# Reconstructed image
aprox_image = np.zeros(image.shape)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(image)
axs[0].set_title("Original Image")

axs[1].imshow(aprox_image)
axs[1].set_title("Reconstructed Image")

# Plot lines with hue gradient colors
for i in range(len(sorted_positions) - 1):
    axs[1].plot([sorted_positions[i][0], sorted_positions[i + 1][0]],
                [sorted_positions[i][1], sorted_positions[i + 1][1]],
                color=colors[i], marker='o')

st.pyplot(fig)

# Calculate the length of the path
length = sum(np.linalg.norm(np.array(sorted_positions[i]) - np.array(sorted_positions[i-1])) for i in range(1, len(sorted_positions)))

st.write("Original path length: ", true_length)
st.write("Length of the path: ", length)
