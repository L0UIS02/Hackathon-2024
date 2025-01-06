from pathlib import Path
from argparse import ArgumentParser

from rich.progress import track
import pandas as pd

from solution.mask_generator import *
from solution.position_finder import *


def compute_amazing_solution(
    part_image_path: Path, gripper_image_path: Path
) -> tuple[float, float, float]:
    """Computes a solution for given part and gripper images.

    Args:
        part_image_path (Path): Path to the part image
        gripper_image_path (Path): Path to the gripper image

    Returns:
        float: the x-coordinate of the solution
        float: the y-coordinate of the solution
        float: the angle of the solution
        """

    img = cv.imread(str(part_image_path))
    assert img is not None, "file could not be read, check with os.path.exists()"

    gripper = cv.imread(str(gripper_image_path))
    assert gripper is not None, "file could not be read, check with os.path.exists()"

    gripper = cvt_gripper(gripper)

    for edge_detection_factor in [1.0, 1.03, 1.06, 1.1]:
        holy_matrix = calculate_holy_matrix(img, edge_detection_factor)

        possible_solutions = find_solutions(holy_matrix, gripper)

        if len(possible_solutions) == 0:
            continue

        improved_solutions = []

        for sol in possible_solutions:
            improved_solutions.append(improve_valid_solution(holy_matrix, gripper, sol[0], sol[1], sol[2]))

        best_solution = improved_solutions[0]
        distance_from_center_best_solution = np.inf
        for sol in improved_solutions:

            distance_from_center = calculate_distance_from_center(holy_matrix.shape[0], holy_matrix.shape[1], sol[0], sol[1])

            if distance_from_center < distance_from_center_best_solution:
                best_solution = sol
                distance_from_center_best_solution = distance_from_center

        result_x = float(best_solution[1])
        result_y = float(best_solution[0])
        result_angle = float(360-best_solution[2])

        return result_x, result_y, result_angle

    return img.shape[1]/2, img.shape[0]/2, 0


def main():
    """The main function of your solution.

    Feel free to change it, as long as it maintains the same interface.
    """

    parser = ArgumentParser()
    parser.add_argument("input", help="input csv file")
    parser.add_argument("output", help="output csv file")
    args = parser.parse_args()

    # read the input csv file
    input_df = pd.read_csv(args.input)

    # compute the solution for each row
    results = []
    for _, row in track(
        input_df.iterrows(),
        description="Computing the solutions for each row",
        total=len(input_df),
    ):
        part_image_path = Path(row["part"])
        gripper_image_path = Path(row["gripper"])
        assert part_image_path.exists(), f"{part_image_path} does not exist"
        assert gripper_image_path.exists(), f"{gripper_image_path} does not exist"
        x, y, angle = compute_amazing_solution(part_image_path, gripper_image_path)
        results.append([str(part_image_path), str(gripper_image_path), x, y, angle])

    # save the results to the output csv file
    output_df = pd.DataFrame(results, columns=["part", "gripper", "x", "y", "angle"])
    output_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
