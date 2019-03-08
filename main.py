from typing import Dict, Generic, TypeVar, NamedTuple, List, Tuple
import math
import time
import sys

from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import stl

from objective_function import f
from plot import plot_f, plot_stl


def orient_stl(mesh: stl.mesh.Mesh) -> List[float]:
    res = optimize.dual_annealing(f, args=[mesh], bounds=[[-math.pi, math.pi], [-math.pi, math.pi]])
    # res = optimize.shgo(f, bounds=[[-math.pi, math.pi], [-math.pi, math.pi]])
    # res = optimize.differential_evolution(f, bounds=[[-math.pi, math.pi], [-math.pi, math.pi]],workers=-1)
    # res = optimize.minimize(fun=f, jac=None, method='BFGS', bounds=[(0, math.pi*2), (0, math.pi*2), (0, 0)], x0=np.array([0, 0, 0]), options={'xatol': 1E-10})
    print(f'f({np.round(res.x, decimals=2)})={f(res.x, mesh)}')
    return res.x


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python main.py STL_FILENAME (orient|plot)')
    else:
        filename = sys.argv[1]
        command = sys.argv[2]
        mesh = stl.mesh.Mesh.from_file(filename)
        if command == 'plot':
            print(f'Plotting function value for {filename}')
            plot_f((100,100), mesh)
        elif command == 'orient':
            theta = orient_stl(mesh)
            print(f'File "{filename}" oriented with angles {theta}')

