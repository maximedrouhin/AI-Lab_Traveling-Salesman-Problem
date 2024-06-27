import copy
import numpy as np
import line_detection_functions as ldf

def get_connection_matrix(image, positions,score_function=lambda x: 1 - x, line_detection_function = ldf.get_ratio_blue_pixels) -> np.ndarray:
    connection_matrix = np.full((len(positions), len(positions)), np.inf, dtype=np.float64)
    for i in range(len(positions)-1):
        for j in range(i+1,len(positions)):
            ratio = line_detection_function(image, positions[i], positions[j])
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

    return best_path

#taking to long to run
def return_paths(connection_matrix, max_value, current_path, current_score):
    if not current_path:  #check if current_path is empty
        return []
    if len(current_path) == connection_matrix.shape[0]:
        return current_path
    #print(len(current_path),current_path)
    paths = []
    for i in range(len(connection_matrix)):
        if i not in current_path:
            new_score = current_score + connection_matrix[current_path[-1]][i]
            if new_score <= max_value:
                new_path = current_path.copy()
                new_path.append(i)
                paths.extend(return_paths(connection_matrix, max_value, new_path, new_score))
    return paths

#taking to long to run
def get_best_path(connection_matrix: np.ndarray,abort: float) -> tuple[list, float]:
    simple_score = 0.01
    paths = []
    while not paths:
        paths = return_paths(connection_matrix, simple_score, [0], 0)
        simple_score = simple_score * 1.1
        if simple_score > abort:
            break
        #print(simple_score)
    return paths


from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing

def tsp_solver1(connection_matrix: np.ndarray, max_time:int = 60) -> list:
    matr = connection_matrix.copy()
    matr[:, 0] = 0
    permutation, distance = solve_tsp_simulated_annealing(matr, max_processing_time=max_time)
    return permutation


def get_valid_path_backcheck(connection_matrix: np.ndarray, max_val: float, zero_matr: bool = True) -> list:

    original_connections = {}
    deleted_connections = {}

    def create_new_valid_connections(valid_connections, next_pos):
        deleted_connections[next_pos] = valid_connections.pop(next_pos)

        one = 0
        for node,connection in valid_connections.items():
            if next_pos in connection:
                connection.remove(next_pos)
            if len(connection) <= 1:
                #print(f"Node {node} has only {len(connection)} connection left")
                if len(connection) == 0:
                    return True
                one += 1
                if one > 2:
                    return True
        return False
    
    def restore_connections(valid_connections, restore_pos):
        for node,connection in valid_connections.items():
            if restore_pos in original_connections[node]:
                valid_connections[node] = [x for x in original_connections[node] if x == restore_pos or x in connection]
        valid_connections[restore_pos] = deleted_connections.pop(restore_pos)
        return None

    def valid_path_searcher(valid_connections, path):
        #print(valid_connections)
        #print(path)
        #print(len(path))
        reamining_options = list(valid_connections[path[-1]])

        if len(valid_connections) == 2:
            if reamining_options:
                path.append(reamining_options[0])
                return path

        #the call to create_new_valid_connections will remove the next_pos from the valid_connections this will edit the dict!
        if create_new_valid_connections(valid_connections, path[-1]):
            restore_connections(valid_connections, path[-1])
            return None

        
        while len(reamining_options) > 0:
                next_pos = reamining_options.pop(0)
                path.append(next_pos)
                next = valid_path_searcher(valid_connections, path)
                if next is not None:
                    return next
                else:
                    path.pop()

        restore_connections(valid_connections, path[-1])
        
        return None


    valid_connections = {}

    # if zero_matr is True, we subtract the minimum value of each row from the row
    if zero_matr:
        for i in range(connection_matrix.shape[0]):
            connection_matrix[i] = connection_matrix[i] - min(connection_matrix[i])

    #get all the valid connections for each point
    for i in range(connection_matrix.shape[0]):
        valid_connections[i] = np.nonzero(connection_matrix[i] < max_val)

    #sort the connections by the value of the connection
    for i in range(connection_matrix.shape[0]):
        argsorted = np.argsort(connection_matrix[i][valid_connections[i]])
        valid_connections[i] = list(valid_connections[i][0][argsorted])
    
    #check if the first has at least one connection
    if len(valid_connections[0]) == 0:
        #raise Exception(f"Point 0 has no valid connections")
        return None
    #check if everyone has at least two connection except the first and last
    one_connection = None
    for i in range(1, connection_matrix.shape[0]):
        if len(valid_connections[i]) == 0:
            #raise Exception(f"Point {i} has no valid connections")
            return None
        elif len(valid_connections[i]) == 1:
            if one_connection is not None:
                #raise Exception(f"Point {i} has only one valid connection, but {one_connection} also has only one valid connection")
                return None
            one_connection = i
    
    #valid_connections = create_new_valid_connections(valid_connections, 0)
    #original_connections = valid_connections.copy()
    original_connections = copy.deepcopy(valid_connections)

    path = valid_path_searcher(valid_connections, [0])
    #if not path:
    #    raise Exception("No valid path found")
    return path



import time

def get_valid_path_backcheck_maxtime(connection_matrix: np.ndarray, max_val: float, zero_matr: bool = True, maxtime: int = 60) -> list:

    original_connections = {}
    deleted_connections = {}
    end_time = time.time() + maxtime

    def create_new_valid_connections(valid_connections, next_pos):
        deleted_connections[next_pos] = valid_connections.pop(next_pos)

        one = 0
        for node,connection in valid_connections.items():
            if next_pos in connection:
                connection.remove(next_pos)
            if len(connection) <= 1:
                #print(f"Node {node} has only {len(connection)} connection left")
                if len(connection) == 0:
                    return True
                one += 1
                if one > 2:
                    return True
        return False
    
    def restore_connections(valid_connections, restore_pos):
        for node,connection in valid_connections.items():
            if restore_pos in original_connections[node]:
                valid_connections[node] = [x for x in original_connections[node] if x == restore_pos or x in connection]
        valid_connections[restore_pos] = deleted_connections.pop(restore_pos)
        return None

    def valid_path_searcher(valid_connections, path):
        #print(valid_connections)
        #print(path)
        #print(len(path))
        reamining_options = list(valid_connections[path[-1]])

        if len(valid_connections) == 2:
            if reamining_options:
                path.append(reamining_options[0])
                return path

        #the call to create_new_valid_connections will remove the next_pos from the valid_connections this will edit the dict!
        if create_new_valid_connections(valid_connections, path[-1]):
            restore_connections(valid_connections, path[-1])
            return None

        
        while len(reamining_options) > 0:
                if time.time() > end_time:
                    return None
                next_pos = reamining_options.pop(0)
                path.append(next_pos)
                next = valid_path_searcher(valid_connections, path)
                if next is not None:
                    return next
                else:
                    path.pop()

        restore_connections(valid_connections, path[-1])
        
        return None


    valid_connections = {}

    # if zero_matr is True, we subtract the minimum value of each row from the row
    if zero_matr:
        for i in range(connection_matrix.shape[0]):
            connection_matrix[i] = connection_matrix[i] - min(connection_matrix[i])

    #get all the valid connections for each point
    for i in range(connection_matrix.shape[0]):
        valid_connections[i] = np.nonzero(connection_matrix[i] < max_val)

    #sort the connections by the value of the connection
    for i in range(connection_matrix.shape[0]):
        argsorted = np.argsort(connection_matrix[i][valid_connections[i]])
        valid_connections[i] = list(valid_connections[i][0][argsorted])
    
    #check if the first has at least one connection
    if len(valid_connections[0]) == 0:
        #raise Exception(f"Point 0 has no valid connections")
        return None
    #check if everyone has at least two connection except the first and last
    one_connection = None
    for i in range(1, connection_matrix.shape[0]):
        if len(valid_connections[i]) == 0:
            #raise Exception(f"Point {i} has no valid connections")
            return None
        elif len(valid_connections[i]) == 1:
            if one_connection is not None:
                #raise Exception(f"Point {i} has only one valid connection, but {one_connection} also has only one valid connection")
                return None
            one_connection = i
    
    #valid_connections = create_new_valid_connections(valid_connections, 0)
    #original_connections = valid_connections.copy()
    original_connections = copy.deepcopy(valid_connections)

    path = valid_path_searcher(valid_connections, [0])
    #if not path:
    #    raise Exception("No valid path found")
    return path


def get_valid_path_backcheck_multiple(connection_matrix: np.ndarray, start:float, stop:float, step:float, zero_matr: bool = True, maxtime: int = 60) -> list:
    maxval = start
    while maxval < stop:
        path = get_valid_path_backcheck_maxtime(connection_matrix, maxval, zero_matr, maxtime)
        if path:
            return path
        maxval += step
    return None