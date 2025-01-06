import cv2 as cv
import numpy as np
from scipy.ndimage import rotate
import math


def reduce_gripper_matrix(gripper_matrix):
    """Cuts off any outside rows or columns that only contain 0.

    Args:
        gripper_matrix (numpy.ndarray): The gripper matrix

    Returns:
        numpy.ndarray: The reduced gripper matrix
        """

    non_zero_rows = np.any(gripper_matrix != 0, axis=1)
    first_non_zero_row = np.argmax(non_zero_rows)
    last_non_zero_row = len(non_zero_rows) - np.argmax(non_zero_rows[::-1]) - 1

    non_zero_columns = np.any(gripper_matrix != 0, axis=0)
    first_non_zero_column = np.argmax(non_zero_columns)
    last_non_zero_column = len(non_zero_columns) - np.argmax(non_zero_columns[::-1]) - 1

    trimmed_matrix = gripper_matrix[first_non_zero_row:last_non_zero_row+1, first_non_zero_column:last_non_zero_column+1]

    return trimmed_matrix


def rotate_gripper(gripper_matrix, angle):
    """Rotates a gripper counterclockwise by a given angle.

    Args:
        gripper_matrix (numpy.ndarray): The gripper matrix
        angle (float): The angle to rotate the gripper by

    Returns:
        numpy.ndarray: The rotated gripper matrix
        """

    rotated = rotate(gripper_matrix, angle, reshape=True, order=1)

    reduce_gripper_matrix(rotated)

    return (rotated > 0.5).astype(int)


def cvt_gripper(gripper_image):
    """Converts a gripper matrix so that it is 1 where the gripper is and 0 elsewhere.

    Args:
        gripper_image (numpy.ndarray): The image of the gripper

    Returns:
        numpy.ndarray: The converted gripper matrix
        """

    return np.where(cv.cvtColor(gripper_image, cv.COLOR_BGR2GRAY) == 0, 0, 1)


def calculate_gripper_hole_overlap(holy_matrix, gripper_matrix, x, y, angle):
    """Counts overlapping pixels of the gripper with holes for a given position and angle.

    Args:
        holy_matrix (numpy.ndarray): The matrix containing the information about the holes
        gripper_matrix (numpy.ndarray): The gripper matrix
        x (int): The x position of the center of the gripper (x-axis is pointing down)
        y (int): The y position of the center of the gripper (y-axis is pointing right)
        angle (float): The angle of the gripper

    Returns:
        int: The number of overlapping pixels of the gripper with holes
        """

    gripper_matrix = rotate_gripper(gripper_matrix, angle)

    grip_rows, grip_cols = gripper_matrix.shape
    comp_rows, comp_cols = holy_matrix.shape

    # Grenzen prüfen
    if x + 1 + (grip_rows - 1) / 2 > comp_rows or y + 1 + (grip_cols - 1) / 2 > comp_cols or x + 1 < grip_rows / 2 or y + 1 < grip_cols / 2:
        # print("Grenzen stimmen nicht.")
        return count_gripper_pixels(gripper_matrix)

    # Prüfe Überschneidung mit Löchern
    x_1 = int(x + 1 - grip_rows / 2 + 0.25)
    y_1 = int(y + 1 - grip_cols / 2 + 0.25)
    x_2 = int(x + 1 + grip_rows / 2 + 0.25)
    y_2 = int(y + 1 + grip_cols / 2 + 0.25)
    # print("Submatrix in den Grenzen: (", x_1, "-", x_2, ", ", y_1, "-", y_2, ")", "  Gripper Matrix: (", grip_rows, ", ", grip_cols, ")", "  Position: (", x, ", ", y, ")")
    sub_matrix = holy_matrix[x_1:x_2, y_1:y_2]

    gripper_pixel_indices = np.argwhere(gripper_matrix == 1)

    overlap = 0

    for (i, j) in gripper_pixel_indices:
        if sub_matrix[i, j] != 0:
            overlap += 1

    # show_three_images(sub_matrix, gripper_matrix, gripper_matrix, "Submatrix", "Gripper", "Gripper")

    return overlap


def calculate_overlap_gradient(holy_matrix, gripper_matrix, x, y, angle, d_angle = 1):
    """Calculates the gradient of the overlap between the gripper and holes.

    Args:
        holy_matrix (numpy.ndarray): The matrix containing the information about the holes
        gripper_matrix (numpy.ndarray): The gripper matrix
        x (int): The x position of the center of the gripper (x-axis is pointing down)
        y (int): The y position of the center of the gripper (y-axis is pointing right)
        angle (float): The angle of the gripper
        d_angle (float): The difference of the angle of the gripper

    Returns:
        int: the first component of the gradient (x)
        int: the first component of the gradient (y)
        int: the third component of the gradient (angle)
        """

    overlap_gradient_x = calculate_gripper_hole_overlap(holy_matrix, gripper_matrix, x+1, y, angle) - calculate_gripper_hole_overlap(holy_matrix, gripper_matrix, x, y, angle)
    overlap_gradient_y = calculate_gripper_hole_overlap(holy_matrix, gripper_matrix, x, y+1, angle) - calculate_gripper_hole_overlap(holy_matrix, gripper_matrix, x, y, angle)
    overlap_gradient_angle = calculate_gripper_hole_overlap(holy_matrix, gripper_matrix, x, y, angle+d_angle) - calculate_gripper_hole_overlap(holy_matrix, gripper_matrix, x, y, angle-d_angle)

    gripper_pixels = count_gripper_pixels(gripper_matrix)

    if abs(overlap_gradient_x) > gripper_pixels:
        overlap_gradient_x = 0

    if abs(overlap_gradient_y) > gripper_pixels:
        overlap_gradient_y = 0

    if abs(overlap_gradient_angle) > gripper_pixels:
        overlap_gradient_angle = 0

    return overlap_gradient_x, overlap_gradient_y, overlap_gradient_angle


def count_gripper_pixels(gripper):
    """Counts the pixels (equal to 1) of the gripper.

    Args:
        gripper (numpy.ndarray): The gripper matrix

    Returns:
        int: The number of pixels of the gripper
        """

    return np.sum(gripper)


def round_up_amount(a):
    """Rounds a number in such a way that its amount is always increased if is not an integer.

    Args:
        a (float): The number to round

    Returns:
        int: The rounded number
        """

    if a > 0:
        return math.ceil(a)
    else:
        return math.floor(a)


def find_local_minimum(holy_matrix, gripper_matrix, x, y, angle, gripper_pixels, optimization_speed, iteration,
                       max_iterations):
    """Finds a local minimum of gripper-hole overlap using a simplified gradient descent method.

    Args:
        holy_matrix (numpy.ndarray): The matrix containing the information about the holes
        gripper_matrix (numpy.ndarray): The gripper matrix
        x (int): The x position of the center of the gripper
        y (int): The y position of the center of the gripper
        angle (float): The angle of the gripper
        gripper_pixels (int): The number of pixels of the gripper
        optimization_speed (int): The speed of the change of the position
        iteration (int): The current iteration
        max_iterations (int): The maximum number of iterations

    Returns:
        int: the x-coordinate of the local minimum that was found
        int: the y-coordinate of the local minimum that was found
        int: the angle of the local minimum that was found
        """

    if iteration >= max_iterations:
        return x, y, angle

    if calculate_gripper_hole_overlap(holy_matrix, gripper_matrix, x, y, angle) == 0:
        return x, y, angle

    grad_x, grad_y, grad_angle = calculate_overlap_gradient(holy_matrix, gripper_matrix, x, y, angle)

    if grad_x == 0 and grad_y == 0 and grad_angle == 0:
        return x, y, angle

    return find_local_minimum(holy_matrix, gripper_matrix,
                              x - round_up_amount(grad_x / (gripper_pixels/100) * optimization_speed),
                              y - round_up_amount(grad_y / (gripper_pixels/100) * optimization_speed),
                              angle - round_up_amount(grad_angle / (gripper_pixels/100) * optimization_speed),
                              gripper_pixels, optimization_speed, iteration + 1, max_iterations)


def find_starting_points(x1, x2, y1, y2, dx, dy, d_angle, distance_to_boundaries, number_of_starting_points = 20,
                         angle_1 = 0, angle_2 = 360, angle_3 = 0, angle_4 = 0):
    """Randomly chooses starting points for the find_local_minimum() function in a given area.

    Args:
        x1 (int): The x-coordinate of the upper left corner of the area of possible starting points
        x2 (int): The x-coordinate of the lower right corner of the area of possible starting points
        y1 (int): The y-coordinate of the upper left corner of the area of possible starting points
        y2 (int): The y-coordinate of the lower right corner of the area of possible starting points
        dx (int): The minimum distance in x-direction of two starting points
        dy (int): The minimum distance in y-direction of two starting points
        d_angle (int): The step width of possible angles for starting points
        distance_to_boundaries (int): The distance of starting points to the boundaries of the area
        number_of_starting_points (int): The number of starting-points to choose (default 20)
        angle_1 (int): The lower bound of the first zone of possible angles (default 0)
        angle_2 (int): The upper bound of the first zone of possible angles (default 360)
        angle_3 (int): The lower bound of the second zone of possible angles (default 0)
        angle_4 (int): The upper bound of the second zone of possible angles (default 0)

    Returns:
        numpy.array: A list of starting points
        """

    starting_points = []

    for i in range(x1+distance_to_boundaries, x2-distance_to_boundaries, dx):
        for j in range(y1+distance_to_boundaries, y2-distance_to_boundaries, dy):
            for k in range(angle_1, angle_2, d_angle):
                starting_points.append((i, j, k))
            for k in range(angle_3, angle_4, d_angle):
                starting_points.append((i, j, k))

    np_starting_points = np.array(starting_points)

    possible_starting_points_count = np_starting_points.shape[0]

    return np_starting_points[np.random.choice(possible_starting_points_count, size=min(number_of_starting_points, possible_starting_points_count), replace=False)]


def find_part_position(holy_matrix):
    """Finds the corners of the rectangle that contains the actual part.

    Args:
        holy_matrix (numpy.ndarray): The matrix containing the information about the holes

    Returns:
        int: the x-coordinate of the upper left corner of the rectangle
        int: the y-coordinate of the upper left corner of the rectangle
        int: the x-coordinate of the lower right corner of the rectangle
        int: the y-coordinate of the lower right corner of the rectangle"""

    rows_without_part = np.all(holy_matrix == 1, axis=1)
    columns_without_part = np.all(holy_matrix == 1, axis=0)

    x1 = np.where(rows_without_part == False)[0][0]
    x2 = np.where(rows_without_part == False)[0][-1]
    y1 = np.where(columns_without_part == False)[0][0]
    y2 = np.where(columns_without_part == False)[0][-1]

    return x1, y1, x2, y2


def find_solutions(holy_matrix, gripper_matrix, dx = 20, dy = 20, d_angle = 5, optimization_speed = 1,
                   number_of_starting_points = 50, distance_to_boundaries_relative_to_area = 1/5):
    """Computes several local minima and chooses valid solutions.

    Args:
        holy_matrix (numpy.ndarray): The matrix containing the information about the holes
        gripper_matrix (numpy.ndarray): The gripper matrix
        dx (int): The minimum distance in x-direction of two starting points (default 20)
        dy (int): The minimum distance in y-direction of two starting points (default 20)
        d_angle (float): The step width of possible angles for starting points (default 5)
        optimization_speed (int): The speed of the change of the position (default 1)
        number_of_starting_points (int): The number of starting-points (default 50)
        distance_to_boundaries_relative_to_area (float): measure for the distance to the boundaries of the starting
            points relative to the distance between the boundaries

    Returns:
        np.array: A list of valid solutions
        """

    holy_rows, holy_cols = holy_matrix.shape
    gripper_rows, gripper_cols = gripper_matrix.shape
    max_gripper_dimension = max(gripper_rows, gripper_cols)

    x1_part, y1_part, x2_part, y2_part = find_part_position(holy_matrix)

    gripper_x, gripper_y = int(max_gripper_dimension / 2) + 1, int(max_gripper_dimension / 2) + 1

    angle_1, angle_2, angle_3, angle_4 = 0, 360, 0, 0

    if gripper_rows > holy_rows or gripper_cols > holy_cols:
        gripper_x = int(gripper_cols / 2) + 1
        gripper_y = int(gripper_rows / 2) + 1
        if gripper_rows < 2*holy_rows and gripper_cols < 2*holy_cols:
            angle_1, angle_2, angle_3, angle_4 = 65, 116, 245, 296
            d_angle = 3
        else:
            angle_1, angle_2, angle_3, angle_4 = 85, 96, 265, 276
            d_angle = 2

    elif gripper_rows > holy_cols or gripper_cols > holy_rows:
        gripper_x = int(gripper_rows / 2) + 1
        gripper_y = int(gripper_cols / 2) + 1
        if gripper_rows < 2*holy_cols and gripper_cols < 2*holy_rows:
            angle_1, angle_2, angle_3, angle_4 = -25, 26, 155, 206
            d_angle = 3
        else:
            angle_1, angle_2, angle_3, angle_4 = -5, 6, 175, 186
            d_angle = 2

    x1, y1 = gripper_x + x1_part, gripper_y + y1_part
    x2, y2 = x2_part - gripper_x, y2_part - gripper_y

    distance_to_boundaries = math.ceil(min(x2-x1, y2-y1) * distance_to_boundaries_relative_to_area)

    starting_points = find_starting_points(x1, x2, y1, y2, dx, dy, d_angle, distance_to_boundaries,
                                           number_of_starting_points, angle_1, angle_2, angle_3, angle_4)

    gripper_pixels = count_gripper_pixels(gripper_matrix)

    possible_solutions = []

    for (x, y, angle) in starting_points:
        local_minimum = find_local_minimum(holy_matrix, gripper_matrix, x, y, angle, gripper_pixels, optimization_speed,
                                           0, 20)
        if calculate_gripper_hole_overlap(holy_matrix, gripper_matrix, local_minimum[0], local_minimum[1],
                                          local_minimum[2]) == 0:
            possible_solutions.append(local_minimum)

    return np.array(possible_solutions)


def calculate_distance_from_center(rows, cols, x, y):
    """Calculates the distance of a solution from the center of a part.

    Args:
        rows (int): The number of rows of the image
        cols (int): The number of columns of the image
        x (int): The x coordinate of the solution
        y (int): The y coordinate of the solution

    Returns:
        float: The distance from the center of the part
        """

    return math.sqrt((x - rows/2)**2 + (y - cols/2)**2)


def improve_valid_solution(holy_matrix, gripper, x, y, angle):
    """Shifts gripper to better, but still valid positions until it is not possible.

    Args:
        holy_matrix (numpy.ndarray): The matrix containing the information about the holes
        gripper (numpy.ndarray): The gripper matrix
        x (int): The x-coordinate of the solution
        y (int): The y-coordinate of the solution
        angle (float): The angle of the solution

    Returns:
        int: The x-coordinate of the improved solution
        int: The y-coordinate of the improved solution
        angle: The angle of the improved solution
        """

    rows, cols = holy_matrix.shape

    if x < int(rows/2 + 0.5):
        dx = 1
    else:
        dx = -1
    if y < int(cols/2 + 0.5):
        dy = 1
    else:
        dy = -1

    while True:
        if calculate_gripper_hole_overlap(holy_matrix, gripper, x+dx, y+dy, angle) == 0 and abs(x-rows/2) > 0.5 and abs(y-cols/2) > 0.5:
            x += dx
            y += dy
            continue
        if calculate_gripper_hole_overlap(holy_matrix, gripper, x+dx, y, angle) == 0 and abs(x-rows/2) > 0.5:
            x += dx
            continue
        if calculate_gripper_hole_overlap(holy_matrix, gripper, x, y+dy, angle) == 0 and abs(y-cols/2) > 0.5:
            y+=dy
            continue
        break

    return x, y, angle