import cv2
import numpy as np
import fast_tsp

def add_circle(image, center, circle):
    image = image.copy()
    image = image.astype(np.uint16)
    y, x = center
    x -= circle.shape[0] // 2
    y -= circle.shape[1] // 2
    image[x:(x+circle.shape[0]), y:(y+circle.shape[1])] += circle
    image = np.where(image > 255, 255, image)
    image = image.astype(np.uint8)
    return image

def delete_overlapping_circles(positions, distance):
    new_positions = []
    for i in range(len(positions)):
        overlapping = False
        for j in range(len(positions)):
            if i != j:
                dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                if dist < distance:
                    overlapping = True
                    break
        if not overlapping:
            new_positions.append(positions[i])
    return np.array(new_positions)

def generate_random_picure():
    
    
    maxsize = 1024
    minsize = 200

    N=100

    minradius = 5
    maxradius = 20


    radius = 7
    delete_distance = 5*radius


    # Define the color of the circles (in BGR format)
    color = (0, 255, 255)  # Yellow color

    while True:
        # Generate a random size for the image between maxsize and minsize
        image_size = np.random.randint(minsize, maxsize+1)

        # Generate random positions and radii for the circles
        positions = np.random.randint(radius, image_size-30, size=(N, 2))+15

        positions = delete_overlapping_circles(positions, delete_distance)
        
        if positions.shape[0] >= 2:
            break


    #solve for shortest path
    distance_matrix = np.zeros((positions.shape[0], positions.shape[0]),dtype=int)
    for i in range(positions.shape[0]):
        for j in range(positions.shape[0]):
            distance_matrix[i, j] = int(np.linalg.norm(positions[i] - positions[j]))

    path = fast_tsp.find_tour(distance_matrix)

    positions = positions[path]

    # Create a black image with the generated size
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    

    #draw the path
    for i in range(positions.shape[0]-1):
        for th in [6,4,2,1]:
            overlay = image.copy()
            alpha = 65/255

            cv2.line(overlay, tuple(positions[i]), tuple(positions[i+1]), (0, 130, 255), th, cv2.LINE_AA)

            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw the circles on the image
    circle = np.load("circle.npy")
    for i in range(positions.shape[0]):
        center = tuple(positions[i])

        image = add_circle(image, center, circle)


    return ((image), (positions))

