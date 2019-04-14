import math
from typing import List, Tuple, Callable

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import stl


def plot_f(resolution: Tuple[float,float], f):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Effect of Rotating Part on Function Value')
    ax.set_xlabel('X-rotation')
    ax.set_ylabel('Y-rotation')
    ax.set_zlabel('Function value')

    x_rotation = np.linspace(-math.pi, math.pi, resolution[0])
    y_rotation = np.linspace(-math.pi, math.pi, resolution[1])
    x_rotation_mesh, y_rotation_mesh = np.meshgrid(x_rotation, y_rotation, indexing='ij')

    f_of_t = np.array([[f([x, y, 0]) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(x_rotation_mesh, y_rotation_mesh)])
    
    print(f'Minimum value of {np.amin(f_of_t)} found')
    s = ax.plot_surface(x_rotation_mesh, y_rotation_mesh, f_of_t)
    plt.show()

def plot_stl(theta: List[float], mesh: stl.mesh.Mesh):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Optimal Orientation of Part')

    mesh = stl.mesh.Mesh(mesh.data.copy())
    mesh.rotate([1, 0, 0], theta[0])
    mesh.rotate([0, 1, 0], theta[1])
    if len(theta) > 2:
        mesh.rotate([0, 0, 1], theta[2])
    stl_polygons = mplot3d.art3d.Poly3DCollection(mesh.vectors)
    stl_polygons.set_facecolor('gold')
    stl_polygons.set_edgecolor('black')
    ax.add_collection3d(stl_polygons)
    scale = mesh.points.ravel()
    ax.auto_scale_xyz(scale, scale, scale)
    plt.show()
