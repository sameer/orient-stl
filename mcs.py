from typing import Dict, Generic, TypeVar, NamedTuple, List
import math

from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from stl import mesh

original_mesh = mesh.Mesh.from_file('stardestroyer-engines.stl')


def f(x):
    new_mesh = mesh.Mesh(original_mesh.data.copy(), remove_empty_areas=True)
    # rotate face to match orientation of the build plane
    roll = new_mesh.rotation_matrix([1, 0, 0], x[0])
    pitch = new_mesh.rotation_matrix([0, 1, 0], x[1])
    yaw = new_mesh.rotation_matrix([0, 0, 1], x[2])
    rotation_matrix = np.dot(np.dot(roll, pitch), yaw)
    new_mesh.rotate_using_matrix(rotation_matrix)
    new_mesh.update_normals()

    # find the z-height of the lowest vertex
    lowest_vertex_z = np.amin(new_mesh.vectors[:,:,2])

    # Faces of all the triangles in 2D
    faces_2d = np.float128(new_mesh.vectors[:,:,:2])
    overhang_face_indices = np.extract((new_mesh.normals[:,2] < 0) * (new_mesh.vectors[:,:,2] > lowest_vertex_z).any(axis=1), np.arange(0, len(faces_2d)))
    faces_2d = np.take(faces_2d, overhang_face_indices, axis=0)
    sides = np.array([faces_2d[:, 0] - faces_2d[:, 1], faces_2d[:, 1] - faces_2d[:, 2], faces_2d[:, 2] - faces_2d[:, 0]])
    lengths = np.linalg.norm(sides, axis=2)
    lengths.sort(axis=0)
    c,b,a = lengths[0], lengths[1], lengths[2]
    heron = (a + (b+c))*(c - (a-b))*(c + (a-b))*(a + (b-c))
    
    value = np.sum(.25 * np.sqrt(np.abs(heron)) * np.sign(heron))
    print(f'Computed f({x})={value}')
    return value


res = optimize.minimize(f, method='L-BFGS-B', bounds=[(0, math.pi*2), (0, math.pi*2), (0, math.pi*2)], x0=np.array([4,4,4]), options={'xatol':1E-10})
print(np.round(res.x, decimals=2))

original_mesh.rotate([1, 0, 0], res.x[0])
original_mesh.rotate([0, 1, 0], res.x[1])
original_mesh.rotate([0, 0, 1], res.x[2])
fig = plt.figure()
axes = mplot3d.Axes3D(fig)
stl_polygons = mplot3d.art3d.Poly3DCollection(original_mesh.vectors)
stl_polygons.set_facecolor('gold')
stl_polygons.set_edgecolor('black')
axes.add_collection3d(stl_polygons)
scale = original_mesh.points.ravel()
axes.auto_scale_xyz(scale, scale, scale)
plt.show()
