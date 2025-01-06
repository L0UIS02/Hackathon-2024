import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from solution.position_finder import rotate_gripper


def show_image(image, caption="image"):
    plt.imshow(image, cmap='gray')
    plt.title(caption), plt.xticks([]), plt.yticks([])


def show_three_images(first_image, second_image, third_image, first_caption="First", second_caption="Second",
                      third_caption="Third"):
    plt.subplot(131), show_image(first_image, first_caption)
    plt.subplot(132), show_image(second_image, second_caption)
    plt.subplot(133), show_image(third_image, third_caption)


def show_image_holymatrix_gripper(image, holy_matrix, gripper, image_caption="Image", holy_caption="Hol(e)y Matrix", gripper_caption="Gripper"):
    img = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.axis('off')
    plt.title(image_caption)
    plt.show()

    plt.subplot(121), plt.imshow(holy_matrix, cmap='gray')
    plt.title(holy_caption), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(gripper, cmap='gray')
    plt.title(gripper_caption), plt.xticks([]), plt.yticks([])


def visualize_result(original_image, gripper_matrix, x, y, angle):
    # Rotieren der gripper_matrix um den gegebenen Winkel
    rotated_gripper = rotate_gripper(gripper_matrix, angle)

    # Berechnung der Eckpunkte der eingefügten Matrix
    gripper_shape = rotated_gripper.shape
    org_shape = original_image.shape

    top_left_x = int(x - gripper_shape[0] // 2)
    top_left_y = int(y - gripper_shape[1] // 2)

    bottom_right_x = top_left_x + gripper_shape[0]
    bottom_right_y = top_left_y + gripper_shape[1]

    # Überprüfung und Anpassung der Grenzen, falls die Matrix über den Rand hinausgeht
    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)
    bottom_right_x = min(org_shape[0], bottom_right_x)
    bottom_right_y = min(org_shape[1], bottom_right_y)

    # Berechnung der Einfügegrenzen für die kleinere Matrix
    insert_top_left_x = max(0, -int(x - gripper_shape[0] // 2))
    insert_top_left_y = max(0, -int(y - gripper_shape[1] // 2))
    insert_bottom_right_x = insert_top_left_x + (bottom_right_x - top_left_x)
    insert_bottom_right_y = insert_top_left_y + (bottom_right_y - top_left_y)

    gripper_matrix_big = np.zeros(org_shape[0:2])
    gripper_matrix_big[top_left_x:bottom_right_x, top_left_y:bottom_right_y] = \
        rotated_gripper[insert_top_left_x:insert_bottom_right_x, insert_top_left_y:insert_bottom_right_y]

    # Einfügen der kleineren Matrix in die größere Matrix
    result = original_image.copy()
    indices = np.argwhere(gripper_matrix_big > -1)

    for (i, j) in indices:
        if gripper_matrix_big[i, j] != 0:
            result[i, j] = [0, 0, 255]

    result = cv.cvtColor(result, cv.COLOR_BGR2RGB)

    plt.imshow(result)
    plt.axis('off')  # Achsen ausblenden
    plt.show()