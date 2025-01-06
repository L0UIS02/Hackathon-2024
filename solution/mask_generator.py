import cv2 as cv
import numpy as np
from scipy.ndimage import label, find_objects


def calculate_holy_matrix(img, factor=1.0):
    """Computes a matrix which is 0 where the part is and 1 where holes or background is.

    Args:
        img (numpy.ndarray): The image of the part
        factor (float): How sensitive the edge detection is (default 1.0)

    Returns:
        numpy.ndarray: A matrix equal to 1 where holes/background is detected and 0 elsewhere
        """

    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Trenne die HSV-Kanäle
    hue_channel, saturation_channel, value_channel = cv.split(hsv_image)

    # Durchschnittliche Gradienten und 0.9- und 0.75-Quantile berechnen
    hue_quantile_gradient_1 = calculate_quantile_gradient(hue_channel, 0.9*factor)
    saturation_quantile_gradient_1 = calculate_quantile_gradient(saturation_channel, 0.9*factor)
    value_quantile_gradient_1 = calculate_quantile_gradient(value_channel, 0.9*factor)

    hue_quantile_gradient_2 = calculate_quantile_gradient(hue_channel, 0.75*factor)
    saturation_quantile_gradient_2 = calculate_quantile_gradient(saturation_channel, 0.75*factor)
    value_quantile_gradient_2 = calculate_quantile_gradient(value_channel, 0.75*factor)

    hue_edges = cv.Canny(hue_channel, hue_quantile_gradient_2, hue_quantile_gradient_1)
    saturation_edges = cv.Canny(saturation_channel, saturation_quantile_gradient_2, saturation_quantile_gradient_1)
    value_edges = cv.Canny(value_channel, value_quantile_gradient_2, value_quantile_gradient_1)

    # Die Farbdimension mit dem höchsten 0.9-Quantil des Gradienten ist wahrscheinlich die beste, schärfste Darstellung und wird ab jetzt verwendet
    edges = value_edges

    if saturation_quantile_gradient_1 > hue_quantile_gradient_1 and saturation_quantile_gradient_1 > value_quantile_gradient_1:
        edges = saturation_edges

    # Den Farbwert-Kanal nur dann verwenden, wenn die durchschnittliche Sättigung über einem Grenzwert liegt (sonst keine verlässliche Information über die Farbe)
    if hue_quantile_gradient_1 > saturation_quantile_gradient_1 and hue_quantile_gradient_1 > value_quantile_gradient_1 and np.mean(
        saturation_channel) > 20:
        edges = hue_edges

    # Kantenmatrix in Matrix nur mit Nullen (Keine Kante) und Einsen (Kante) umwandeln
    edges = np.where(edges == 255, 1, 0)

    # Kanten verdicken
    edges = set_ones_within_radius(edges, 1)

    inverted_edges = np.where(edges == 1, 0, 1)

    labeled_areas, num_features = label(inverted_edges)

    # Größten Bereich herausfinden
    slices = find_objects(labeled_areas)
    sizes = [np.sum(labeled_areas[s] == (i + 1)) for i, s in enumerate(slices)]
    max_index = np.argmax(sizes)
    largest_region_label = max_index + 1

    holy_matrix = np.where(labeled_areas == largest_region_label, 0, 1)

    # Wenn mehr als die Hälfte der Pixel des Teils schwarz (Wert = 0) ist, ist das "Teil" wahrscheinlich der Hintergrund
    while check_for_background(holy_matrix, img):
        # print("Bedingung erfüllt")
        max_index = np.argmax(sizes)
        sizes[max_index] = -np.inf
        next_biggest_index = np.argmax(sizes)
        next_largest_region_label = next_biggest_index + 1
        holy_matrix = np.where(labeled_areas == next_largest_region_label, 0, 1)

    # Wenn alle drei oder mehr Ecken zum Teil gehören, wurde wahrscheinlich der Hintergrund als Teil erkannt:
    if (holy_matrix[0, 0] + holy_matrix[0, -1] +
        holy_matrix[-1, 0] + holy_matrix[-1, -1] < 2):
        sizes[max_index] = -np.inf
        next_biggest_index = np.argmax(sizes)
        next_largest_region_label = next_biggest_index + 1
        holy_matrix = np.where(labeled_areas == next_largest_region_label, 0, 1)

    return holy_matrix


def calculate_quantile_gradient(image, quantile_value=0.9):
    """calculates the quantile of the gradient of a grayscale image.

    Args:
        image (numpy.ndarray): The (grayscale) image
        quantile_value (float): The quantile of the gradient (default 0.9)

    Returns:
        float: the quantile of the gradient
        """

    sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_magnitude = gradient_magnitude[gradient_magnitude != 0]
    quantile_gradient = np.quantile(gradient_magnitude, quantile_value)
    return quantile_gradient


def set_ones_within_radius(matrix, radius):
    """Sets all elements of a matrix to 1 in a given radius around all elements containing 1.

    Args:
        matrix (numpy.ndarray): The matrix
        radius (int): The radius in which to set all elements to 1

    Returns:
        numpy.ndarray: the new matrix
        """

    result = matrix.copy()

    ones_indices = np.argwhere(matrix == 1)

    for (i, j) in ones_indices:
        row_min = max(i - radius, 0)
        row_max = min(i + radius + 1, matrix.shape[0])
        col_min = max(j - radius, 0)
        col_max = min(j + radius + 1, matrix.shape[1])

        result[row_min:row_max, col_min:col_max] = 1

    return result


def check_for_background(holy_matrix, original_image):
    """Checks if the supposed part is indeed likely background (equal to 0).

    Args:
        holy_matrix (numpy.ndarray): The matrix showing the holes of the part
        original_image (numpy.ndarray): The original image of the part

    Returns:
        bool: the supposed part is likely background
        """

    gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

    part_indices = np.argwhere(holy_matrix == 0)
    black_pixel_count = 0
    pixel_count = 0

    for (i, j) in part_indices:
        pixel_count += 1
        if gray_image[i, j] == 0:
            black_pixel_count += 1

    if black_pixel_count > len(part_indices) * 0.5:
        return True
    return False