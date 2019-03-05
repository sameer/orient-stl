from typing import Dict, Generic, TypeVar, NamedTuple, List
import math
import time

from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from stl import mesh

FILE_NAME = 'death star.stl'

original_mesh: mesh.Mesh = mesh.Mesh.from_file(FILE_NAME)


def f(x):
    start = time.time()
    # need to copy, rotations in-place lose significant precision
    new_mesh = mesh.Mesh(original_mesh.data.copy(), remove_empty_areas=True)

    # rotate face to match orientation of the build plate
    roll = new_mesh.rotation_matrix([1, 0, 0], x[0])
    pitch = new_mesh.rotation_matrix([0, 1, 0], x[1])
    yaw = new_mesh.rotation_matrix([0, 0, 1], x[2])

    rotation_matrix = roll @ pitch @ yaw
    new_mesh.rotate_using_matrix(rotation_matrix)
    new_mesh.update_normals()

    # z-height of the lowest vertex
    lowest_vertex_z = np.amin(new_mesh.vectors[:, :, 2])

    # faces of all the triangles in 2D, precision bumped to 128-bit where possible
    faces_2d = np.float128(new_mesh.vectors[:, :, :2])
    # find overhangs that aren't bottom-surfaces
    overhang_face_indices = np.extract((new_mesh.normals[:, 2] < 0) * (new_mesh.vectors[:, :, 2] > lowest_vertex_z).any(axis=1), np.arange(0, len(faces_2d)))
    # prune faces to overhangs
    faces_2d = np.take(faces_2d, overhang_face_indices, axis=0)
    # sides as vectors
    sides = np.array([faces_2d[:, i] - faces_2d[:, i+1] for i in range(-1, 2)])
    # vector magnitude
    side_lengths = np.linalg.norm(sides, axis=2)

    # compute area using numerically stable Heron's formula
    side_lengths.sort(axis=0)
    c, b, a = side_lengths[0], side_lengths[1], side_lengths[2]
    heron = .25 * np.sqrt(np.abs((a + (b+c))*(c - (a-b))
                                 * (c + (a-b))*(a + (b-c))))
    # Rotations produce some triangles that aren't triangles (sum of smaller 2 side lengths < largest side length)
    # which, when used in Heron's formula, result in sqrt(negative number).
    #
    # Though they should be treated as having zero-area, I treat them as sqrt(abs(number)) * sign(number)
    # so that the objective function may be pseudo-convex (right terminology?).
    heron *= np.sign(heron)

    # sum with pairwise summation, which should keep precision error low
    value = np.sum(heron)

    finish = time.time()
    print(f'Computed f({x})={value} in {round((finish-start)*1000)} ms')
    return value

#res = optimize.differential_evolution(f, bounds=[[-math.pi, math.pi], [-math.pi, math.pi], [0, 0]])
res = optimize.minimize(f, method='L-BFGS-B', bounds=[(
    0, math.pi*2), (0, math.pi*2), (0, 0)], x0=np.array([0, 0, 0]), options={'xatol': 1E-10})
print(np.round(res.x, decimals=2))

original_mesh.rotate([1, 0, 0], res.x[0])
original_mesh.rotate([0, 1, 0], res.x[1])
original_mesh.rotate([0, 0, 1], res.x[2])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Effect of Rotating {FILE_NAME} on Function Value')
ax.set_xlabel('X-rotation')
ax.set_ylabel('Y-rotation')
ax.set_zlabel('Function value')

resolution = [25,25]

x_rotation = np.linspace(-math.pi, math.pi, resolution[0])
y_rotation = np.linspace(-math.pi, math.pi, resolution[1])
x_rotation_mesh, y_rotation_mesh = np.meshgrid(x_rotation, y_rotation, indexing='ij')


f_of_x = np.array([[f([x, y, 0]) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(x_rotation_mesh, y_rotation_mesh)])



s = ax.plot_surface(x_rotation_mesh, y_rotation_mesh, f_of_x)

ax.scatter(res.x[0], res.x[1], f(res.x), c='red')

stl_polygons = mplot3d.art3d.Poly3DCollection(original_mesh.vectors)
stl_polygons.set_facecolor('gold')
stl_polygons.set_edgecolor('black')
# ax.add_collection3d(stl_polygons)
#scale = original_mesh.points.ravel()
#ax.auto_scale_xyz(scale, scale, scale)
# plt.colorbar(s)
plt.show()
