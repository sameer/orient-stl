from typing import Tuple, List, Callable, Dict
from collections import defaultdict
import math

from scipy.optimize import OptimizeResult, minimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def is_cube(bounds_range):
    return (bounds_range == bounds_range[0]).all()

def denormalize_point(bounds, hyper):
    bounds_range = bounds[:, 1] - bounds[:, 0]
    return hyper * bounds_range + bounds[:, 0]


def remove_potentially_optimal(rectangles, fmin: float, epsilon: float):
    fmin_is_zero = np.allclose(fmin, 0)
    sorted_sizes = sorted(rectangles.keys())
    fmin_by_size = [min(rectangle[2] for rectangle in rectangles[size]) for size in sorted_sizes]
    potentially_optimal = {}

    for j, (size, fmin_for_size) in enumerate(zip(sorted_sizes, fmin_by_size)):
        rectangles_same_size = rectangles[size]

        finalists = []
        # 3.3 (6)
        candidates = filter(lambda j: rectangles_same_size[j][2] == fmin_for_size, range(len(rectangles_same_size)))

        for j in candidates:
            candidate = rectangles_same_size[j]

            if j < len(fmin_by_size) - 1:
                # 3.3 (8) or (9)
                minimum_larger_diff = min((fmin_for_larger_size - candidate[2]) / (larger_size -  size) for larger_size, fmin_for_larger_size in zip(sorted_sizes[j+1:], fmin_by_size[j+1:]))
                if not fmin_is_zero and epsilon > (fmin - candidate[2]) / abs(fmin) + size/abs(fmin) * minimum_larger_diff:
                    continue
                elif fmin_is_zero and candidate[2] > size * minimum_larger_diff:
                    continue
                # 3.3 (7)
                if j > 0:
                    maximum_smaller_diff = max((candidate[2] - fmin_for_smaller_size) / (size - smaller_size) for smaller_size, fmin_for_smaller_size in zip(sorted_sizes[:j], fmin_by_size[:j]))
                    if minimum_larger_diff <= 0 or np.allclose(minimum_larger_diff, maximum_smaller_diff) or minimum_larger_diff > maximum_smaller_diff:
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

def size(bounds):
    bounds_range = bounds[:,1] - bounds[:,0]
    bounds_range.sort()
    return np.linalg.norm(bounds_range)
    # log3 = math.log(3)
    # return np.linalg.norm(np.round(np.log(bounds[:, 1] - bounds[:, 0]) / log3))


def split(rectangles, rectangle, fun, args, fun_bounds):    
    _, xmin, fmin = rectangle
    fev = 0

    bounds, center, _ = rectangle
    dimensions = len(bounds)
    bounds_range = bounds[:,1] - bounds[:, 0]
    cube = is_cube(bounds_range)

    if cube:  # Split in all dimensions if cube
        dei = np.diagflat(bounds_range / 3)
    else:  # Split in the largest dimension
        splitting_dim = np.argmax(bounds_range)
        dei = np.zeros((1, dimensions))
        dei[0][splitting_dim] = (bounds[splitting_dim, 1] - bounds[splitting_dim, 0]) / 3

    ci_dei = np.stack((center - dei, center + dei), axis=1)
    f_ci_dei = np.apply_along_axis(lambda x: fun(denormalize_point(fun_bounds, x), *args), 2, ci_dei)
    fev += 2*len(dei)

    f_ci_dei_min_index = tuple(f_ci_dei.argmin(axis=0))
    if f_ci_dei[f_ci_dei_min_index] < fmin:
        xmin = ci_dei[f_ci_dei_min_index]
        fmin = f_ci_dei[f_ci_dei_min_index]
    # else:
    #     print(f_ci_dei[f_ci_dei_min_index])


    wi_indices = np.argmin(f_ci_dei, axis=1)
    wi_values = f_ci_dei[np.arange(0, len(f_ci_dei)), wi_indices]
    
    # Dimensions, ordered from smallest to largest by best wi value
    best_wi_indices = np.argsort(wi_values)

    # best_wi_values = wi_values[best_wi_indices]
    if cube:
        best_deis = bounds_range[best_wi_indices] / 3
    else:
        best_deis = bounds_range[best_wi_indices + splitting_dim] / 3
    # Original rectangle
    prev_rectangle = rectangle

    # Split the hypercube
    splits = 0
    for wi_index, dei in zip(best_wi_indices, best_deis):
        prev_bounds, prev_center, prev_f_center = prev_rectangle
        for i, j in zip([0, 2, 1], [0, 1, -1]): # left, right, center
            bounds_i = prev_bounds.copy()
            if cube:
                bounds_i[wi_index, 0] = i*dei
                bounds_i[wi_index, 1] = (i+1)*dei
            else:
                bounds_i[wi_index + splitting_dim, 0] = i*dei
                bounds_i[wi_index + splitting_dim, 1] = (i+1)*dei
            if j != -1:
                rectangles[size(bounds_i)].append((bounds_i, ci_dei[wi_index][j], f_ci_dei[wi_index][j]))
            else:
                prev_rectangle = (bounds_i, prev_center, prev_f_center)
            splits += 1
    
    rectangles[size(prev_rectangle[0])].append(prev_rectangle)
    return (xmin, fmin, fev)


def initialize(rectangles, fun, args, bounds):
    dimensions = len(bounds)
    # Center
    c1 = np.ones(dimensions) / 2
    # Bounds
    b1 = np.stack((np.zeros(dimensions), np.ones(dimensions)), axis=1)
    # Value
    f_c1 = fun(denormalize_point(bounds, c1), *args)

    return split(rectangles, (b1, c1, f_c1), fun, args, bounds)
    
    # for i, rect in rectangles.items():
    #     print(f'i = {i}: {rect}')


def plot_rectangles(rectangles):
    fig, ax = plt.subplots(1)
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
        print(f'Iteration {it} f({xmin})={fmin} with fev={fev}')
        print(f'Potentially optimal: {potentially_optimal}')
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
    plot_rectangles(rectangles)

    return OptimizeResult(fun=fmin, x=denormalize_point(bounds, xmin))

# print(denormalize_point(np.array([[-10, 10], [-10, 10]]), np.array([0.5, 0.5])))
# assert (denormalize_point(np.array([[-10, 10], [-10, 10]]), np.array([0.5, 0.5])) == np.array([0,0])).all()

print(minimize(lambda x: np.sum(x**2), [0, 0],
               method=direct, bounds=np.array([[-10, 10], [-10, 10]]), options=dict(maxit=5)))
