# Implementation of the dividing rectangle global optimization algorithm.
# Based on: https://ctk.math.ncsu.edu/Finkel_Direct/DirectUserGuide_pdf.pdf
from typing import Tuple, List, Callable, Dict
from collections import defaultdict
import math

from scipy.optimize import OptimizeResult
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def is_cube(bounds_range):
    return np.allclose(bounds_range, bounds_range[0])

def denormalize_point(bounds, hyper):
    bounds_range = bounds[:, 1] - bounds[:, 0]
    return hyper * bounds_range + bounds[:, 0]


def remove_potentially_optimal(rectangles, fmin: float, epsilon: float):
    fmin_is_zero = np.allclose(fmin, 0)
    sorted_sizes = sorted(rectangles.keys())
    fmin_by_size = [min(rectangle[2] for rectangle in rectangles[size]) for size in sorted_sizes]
    potentially_optimal = {}

    for i, (size, fmin_for_size) in enumerate(zip(sorted_sizes, fmin_by_size)):
        rectangles_same_size = rectangles[size]
        finalists = []

        # 3.3 (6)
        candidates = filter(lambda j: rectangles_same_size[j][2] == fmin_for_size, range(len(rectangles_same_size)))

        if i < len(fmin_by_size) - 1:
            larger = [(fmin_for_larger_size / (larger_size - size), larger_size - size) for larger_size, fmin_for_larger_size in zip(sorted_sizes[i+1:], fmin_by_size[i+1:])]
            minimum_larger, minimum_larger_diff = min(larger, key=lambda x: x[0])
            if i > 0:
                smaller = [(fmin_for_smaller_size / (size - smaller_size), size - smaller_size) for smaller_size, fmin_for_smaller_size in zip(sorted_sizes[:i], fmin_by_size[:i])]
                minimum_smaller, minimum_smaller_diff = max(smaller, key=lambda x: x[0])

        for j in candidates:
            fcandidate = rectangles_same_size[j][2]

            if i < len(fmin_by_size) - 1:
                # 3.3 (8) or (9)
                minimum_larger_diff = minimum_larger - fcandidate / minimum_larger_diff
                if not fmin_is_zero and epsilon > (fmin - fcandidate) / abs(fmin) + size/abs(fmin) * minimum_larger_diff:
                    continue
                elif fmin_is_zero and fcandidate > size * minimum_larger_diff:
                    continue
                # 3.3 (7)
                if i > 0:
                    maximum_smaller_diff = fcandidate / minimum_smaller_diff - minimum_smaller
                    if minimum_larger_diff <= 0 or np.allclose(minimum_larger_diff, maximum_smaller_diff) or minimum_larger_diff < maximum_smaller_diff:
                        continue
            finalists.append(j)
        
        if len(finalists) != 0:
            potentially_optimal[size] = finalists

    for size, finalists in potentially_optimal.items():
        finalist_values = []
        for j in reversed(finalists):
            finalist_values.append(rectangles[size].pop(j))
        potentially_optimal[size] = finalist_values
        if len(rectangles[size]) == 0:
            del rectangles[size]
    return potentially_optimal

def bounds_size(bounds):
    bounds_range = bounds[:,1] - bounds[:,0]
    # The reason I sort here is so that error due to floating point precision doesn't increase
    # the number of dictionary keys by too much
    bounds_range.sort()
    return np.linalg.norm(bounds_range)


def split(rectangles, rectangle, fun, args, fun_bounds):    
    _, xmin, fmin = rectangle
    fev = 0

    bounds, center, _ = rectangle
    dimensions = len(bounds)
    bounds_range = bounds[:,1] - bounds[:, 0]

    if is_cube(bounds_range):  # Split in all dimensions if cube
        splitting_offset = 0
        dei = np.diagflat(bounds_range / 3)
    else:  # Split in the largest dimension
        splitting_offset = np.argmax(bounds_range)
        dei = np.zeros((1, dimensions))
        dei[0][splitting_offset] = (bounds[splitting_offset, 1] - bounds[splitting_offset, 0]) / 3

    ci_dei = np.stack((center - dei, center + dei), axis=1)
    f_ci_dei = np.apply_along_axis(lambda x: fun(denormalize_point(fun_bounds, x), *args), 2, ci_dei)
    fev += 2*len(dei)

    f_ci_dei_min_index = tuple(f_ci_dei.argmin(axis=0))
    if f_ci_dei[f_ci_dei_min_index] < fmin:
        xmin = ci_dei[f_ci_dei_min_index]
        fmin = f_ci_dei[f_ci_dei_min_index]


    wi_indices = np.argmin(f_ci_dei, axis=1)
    wi_values = f_ci_dei[np.arange(0, len(f_ci_dei)), wi_indices]
    
    # Dimensions, ordered from smallest to largest by best wi value
    best_wi_indices = np.argsort(wi_values)

    best_deis = bounds_range[best_wi_indices + splitting_offset] / 3
    # Original rectangle
    prev_rectangle = rectangle

    # Split the rectangle
    for wi_index, dei in zip(best_wi_indices, best_deis):
        prev_bounds, prev_center, prev_f_center = prev_rectangle
        for i, j in zip([0, 2, 1], [0, 1, -1]): # left, right, center
            bounds_i = prev_bounds.copy()
            bounds_i[wi_index + splitting_offset, 0] = i*dei
            bounds_i[wi_index + splitting_offset, 1] = (i+1)*dei
            if j != -1:
                rectangles[bounds_size(bounds_i)].append((bounds_i, ci_dei[wi_index][j], f_ci_dei[wi_index][j]))
            else:
                prev_rectangle = (bounds_i, prev_center, prev_f_center)
    
    rectangles[bounds_size(prev_rectangle[0])].append(prev_rectangle)
    return (xmin, fmin, fev)


def initialize(rectangles, fun, args, bounds):
    dimensions = len(bounds)
    # Center of a unit hypercube
    c1 = np.ones(dimensions) / 2
    # Unit hypercube's bounds (0 vector, 1 vector)
    b1 = np.stack((np.zeros(dimensions), np.ones(dimensions)), axis=1)
    # Evaluate at the center
    f_c1 = fun(denormalize_point(bounds, c1), *args)

    xmin, fmin, fev = split(rectangles, (b1, c1, f_c1), fun, args, bounds)
    fev += 1
    return (xmin, fmin, fev)


# Debugging function to view the rectangles made by DIRECT for the first two dimensions
def plot_rectangles(rectangles):
    _, ax = plt.subplots(1)
    boxes = []
    points = []
    count = 0
    for rectangle_list in rectangles.values():
        for rectangle in rectangle_list:
            bounds, center, _ = rectangle
            bounds_range = bounds[:,1] - bounds[:,0]
            corner = center - bounds_range / 2
            boxes.append(Rectangle(tuple(corner[:2]), bounds_range[0], bounds_range[1]))
            points.append(center[:2])
            count += 1
    points = np.array(points)
    pc = PatchCollection(boxes, edgecolor='black', alpha=.5)
    ax.add_collection(pc)
    ax.scatter(points[:, 0], points[:, 1], c='red', marker='.')
    print(f'Plotting {count} rectangles')
    plt.show()

def direct(fun: Callable[[List[float]], float], x0, bounds: List[List[float]], args=(), maxit: int = None, maxfev: int = None, epsilon=1E-4, **options) -> OptimizeResult:
    rectangles: Dict[int, List[Tuple[List[Tuple[float, float]],List[float], float]]] = defaultdict(list)
    
    xmin, fmin, fev = initialize(rectangles, fun, args, bounds)

    it = 0
    
    while True:
        if maxfev is not None and fev >= maxfev:
            break
        if maxit is not None and it >= maxit:
            break
        potentially_optimal = remove_potentially_optimal(rectangles, fmin, epsilon)
        if len(potentially_optimal) == 0:
            break

        for rectangle_list in potentially_optimal.values():
            for rectangle in rectangle_list:
                split_xmin, split_fmin, split_fev = split(rectangles, rectangle, fun, args, bounds)
                if split_fmin < fmin:
                    xmin = split_xmin
                    fmin = split_fmin
                fev += split_fev
        it += 1
        print(f'Iteration {it} f({xmin})={fmin} with fev={fev}')
    # plot_rectangles(rectangles)
    return OptimizeResult(fun=fmin, x=denormalize_point(bounds, xmin))
