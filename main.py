from typing import List, Callable, Tuple
import math
import sys
import time

from scipy import optimize
import numpy as np
import stl

from objective_function import build_f
from plot import plot_f, plot_stl
from direct import direct


def orient_stl(f: Callable[[List[float]],float], debug: bool) -> Tuple[List[float],float]:
    res: Tuple[List[float],float] = optimize.minimize(f, [0, 0], method=direct, bounds=np.array([[-math.pi, math.pi], [-math.pi, math.pi]]), options=dict(maxfev=1000))
    # Other methods were considered and are shown below. The last finds a local minimizer.
    # res = optimize.dual_annealing(f, bounds=[[-math.pi, math.pi], [-math.pi, math.pi]])
    # res = optimize.shgo(f, bounds=[[-math.pi, math.pi], [-math.pi, math.pi]])
    # res = optimize.differential_evolution(f, bounds=[[-math.pi, math.pi], [-math.pi, math.pi]],workers=-1)
    # res = optimize.minimize(fun=f, jac=None, method='BFGS', bounds=[(0, math.pi*2), (0, math.pi*2), (0, 0)], x0=np.array([0, 0, 0]), options={'xatol': 1E-10})
    return (res.x, res.fun)


if __name__ == '__main__':
    debug = False
    if len(sys.argv) != 3:
        print('Usage: python main.py STL_FILENAME (orient|plot|orientplot)')
    else:
        filename = sys.argv[1]
        command = sys.argv[2]
        mesh = stl.mesh.Mesh.from_file(filename)
        f = build_f(mesh, debug)
        start = time.time()
        if command == 'plot':
            print(f'Plotting function value for {filename}')
            plot_f((100,100), f)
        elif command == 'orient':
            theta, value = orient_stl(f, debug)
            finish = time.time()
            print(f'File "{filename}" oriented with angles {np.round(theta, decimals=2)} and value {value} in {round((finish-start)*1000)} ms')
        elif command == 'orientplot':
            theta, value = orient_stl(f, debug)
            finish = time.time()
            print(f'File "{filename}" oriented with angles {np.round(theta, decimals=2)} and value {value} in {round((finish-start)*1000)} ms') 
            plot_stl(theta, mesh)

