import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import circle_detection_functions
import line_detection_functions
from functools import partial
import colorsys

# Initialize the detection functions
circle_detecter_function = partial(circle_detection_functions.get_yellow_circles_cv2)
line_detecter_function = partial(line_detection_functions.get_next_pos)

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

# Detect the circles
detected_positions = list(circle_detecter_function(image))

# Detect the green circle
image_cp = image.copy().astype('int32')
green_matrix = image_cp[:, :, 1] - image_cp[:, :, 0] - image_cp[:, :, 2] - 100
green_matrix = green_matrix.clip(min=0)

all_green_positions = np.nonzero(green_matrix)
average_green_position = np.mean(all_green_positions, axis=1)
start_pos = (int(average_green_position[1]), int(average_green_position[0]))

# Detect the lines
remaining_positions = detected_positions.copy()
current_pos = start_pos
sorted_positions = [current_pos]
while remaining_positions:
    next_pos = line_detecter_function(image, current_pos, remaining_positions)
    current_pos = remaining_positions.pop(next_pos)
    sorted_positions.append(current_pos)

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
