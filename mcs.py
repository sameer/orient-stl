from typing import Dict, Generic, TypeVar, NamedTuple, List
import math

from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from stl import mesh

m = mesh.Mesh.from_file('battery mount.stl')

def f(x):
    msh = mesh.Mesh(m.data.copy(), remove_empty_areas=True)
    # rotate face to match orientation of the build plane
    roll = msh.rotation_matrix([1, 0, 0], x[0])
    pitch = msh.rotation_matrix([0, 1, 0], x[1])
    yaw = msh.rotation_matrix([0, 0, 1], x[2])
    rotation_matrix = np.dot(np.dot(roll, pitch), yaw)
    msh.rotate_using_matrix(rotation_matrix)
    msh.update_normals()
    acc: float = 0.0

    lowest_vertex_z = np.amin(msh.vectors[:,2])
    print(lowest_vertex_z)

    # faces_2d = msh.vectors[:,:,:2]
    # sides = np.array([faces_2d[:, 0] - faces_2d[:, 1], faces_2d[:, 1] - faces_2d[:, 2], faces_2d[:, 2] - faces_2d[:, 0]])
    # lengths = np.linalg.norm(sides, axis=2) * (msh.normals[:,2] < 0) #* (msh.vectors[:,:,2] > lowest_vertex_z).all(axis=1)
    # lengths.sort(axis=0)
    # c,b,a = lengths[0], lengths[1], lengths[2]
    # heron = (a + (b+c))*(c - (a-b))*(c + (a-b))*(a + (b-c))
    # acc = np.sum(.25 * np.sqrt(np.abs(heron)) * np.sign(heron))
    # print(f'Shapes are {faces_2d.shape}, {sides.shape}, {lengths.shape}, {heron.shape}')
    # print(f'Solved for angles {x} with acc {acc}')
    # return acc

    for i, face in enumerate(msh.vectors):
        if len(face) != 3:
            raise Exception('Encountered non-triangular face')
        
        # if face[0] in lowest_vertices and face[1] in lowest_vertices and face[2] in lowest_vertices:
        #     continue
        if msh.normals[i,2] < 0:
            # Calculate area
            face_2d = face[:,:2]
            sides = np.array([face_2d[0] - face_2d[1], face_2d[1] - face_2d[2], face_2d[2] - face_2d[0]])
            lengths = np.linalg.norm(sides, axis=1)
            # lengths = np.sqrt(np.sum(sides**2, axis=1))
            lengths.sort()
            c, b, a = (lengths[0], lengths[1], lengths[2])
            heron = (a + (b+c))*(c - (a-b))*(c + (a-b))*(a + (b-c))
            # if heron < 0:
            #     print(f'Heron {heron} is negative with sides {lengths} for angle {x}')
            acc += .25 * math.sqrt(abs(heron)) * np.sign(heron)
    return acc


# print(f([0,90,0]))
res = optimize.minimize(f, method='L-BFGS-B', bounds=[(0, math.pi*2), (0, math.pi*2), (0, math.pi*2)], x0=np.array([4.3,4,4]), options={'xatol':1E-10})
print(np.round(res.x, decimals=2))

m.rotate([1, 0, 0], res.x[0])
m.rotate([0, 1, 0], res.x[1])
m.rotate([0, 0, 1], res.x[2])
fig = plt.figure()
axes = mplot3d.Axes3D(fig)
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))
scale = m.points.ravel()
axes.auto_scale_xyz(scale, scale, scale)
