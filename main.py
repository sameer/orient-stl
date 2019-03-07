from typing import Dict, Generic, TypeVar, NamedTuple, List, Tuple
import math
import time

from scipy import optimize
from scipy.special import expit
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from stl import mesh

FILE_NAME = 'death star.stl'

original_mesh: mesh.Mesh = mesh.Mesh.from_file(FILE_NAME)


def f(t):
    start = time.time()
    # need to copy, rotations in-place lose significant precision
    vectors = original_mesh.vectors.copy()

    # rotate face to match orientation of the build plate
    roll = original_mesh.rotation_matrix([1, 0, 0], t[0])
    pitch = original_mesh.rotation_matrix([0, 1, 0], t[1])
    yaw = original_mesh.rotation_matrix([0, 0, 1], 0)

    rotation_matrix = roll @ pitch @ yaw
    # vectors = vectors * rotation_matrix
    for i in range(3):
        vectors[:, i] = vectors[:, i] @ rotation_matrix
    a = vectors[:, 1] - vectors[:, 0]
    b = vectors[:, 2] - vectors[:, 0]
    normal_z_components = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]

    # z-height of the lowest vertex
    z_min = np.amin(vectors[:, :, 2])

    # faces of all the triangles in 2D
    faces_2d = vectors[:, :, :2]

    # prune faces to overhangs
    # sides as vectors
    sides = np.array([faces_2d[:, i] - faces_2d[:, i+1] for i in range(-1, 2)])
    # vector magnitude
    side_lengths = np.linalg.norm(sides, axis=2)

    # compute area using numerically stable Heron's formula
    side_lengths.sort(axis=0)
    c, b, a = side_lengths[0], side_lengths[1], side_lengths[2]
    heron = (a + (b+c))*(c - (a-b)) * (c + (a-b))*(a + (b-c))
    # Rotations produce some triangles that aren't triangles (sum of smaller 2 side lengths < largest side length)
    # which, when used in Heron's formula, result in sqrt(negative number).
    #
    # Though they should be treated as having zero-area, I treat them as sqrt(abs(number)) * sign(number)
    # so that the objective function may be pseudo-convex (right terminology?).
    heron_sign = np.sign(heron)

    # Add a bias to emphasize overhang surfaces.
    # 
    # The function computes the total area of the 2D projections of the part. To minimize overhang area,
    # there are several possible treatments:
    # 1. Set overhang area to be positive and top surface area to be negative. However, for some 
    #    symmetric cases like a cube, this will always be 0.
    # 2. Set overhang area to be positive and top surface area to be 0. This is ideal using the unit step function
    #    but results in discontinuous derivatives.
    # 
    # I use a variation on #2: a unit-step approximation. This perturbs the function appropriately and
    # keeps all partial derivatives continuous.
    # As a precautionary measure, I multiply the z-component of the normals by 100 to sharpen the decision boundary.
    overhang_bias = np.tanh(-100*normal_z_components)/2 + .5


    # Add a bias to prefer bottom surfaces.
    # 
    # A triangle with all three points at the lowest vertex height is perpendicular to the build plate. Since it 
    # is also the lowest face of the part, it lies flat on the build plate and is not an overhang. The function
    # should take this into account.
    #
    # 1. Add an if statement to set the area to 0 if a triangle's vertices are all at the minimum z-height. Again,
    #    like with the overhang bias, this results in discontinuous derivatives. For a non-derivative method, the 
    #    function would have no gradient to suggest that it is better to orient a part with big, flat surfaces on 
    #    the bottom.
    #
    # I use a variation on #1: the hyperbolic tangent of the sum of the z-coordinates' distance from z-min.
    # As all three coordinates approach 0, the value approaches 0. 
    # As a precautionary measure, I multiply the sum by 100 to sharpen the decision boundary.
    bottom_face_bias = np.tanh((np.sum(vectors[:, :, 2] ,axis=1) - 3*z_min)*100)

    np.abs(heron, out=heron)
    np.sqrt(heron, out=heron)
    # print(vectors[0:1, :, 2])
    # print(heron)
    # print(overhang_bias)
    # print(bottom_face_bias)
    heron *= overhang_bias
    heron *= bottom_face_bias
    heron /= 4
    heron *= heron_sign

    # sum with pairwise summation, which should keep precision error low
    value = np.sum(heron)

    finish = time.time()
    print(f'Computed f({t})={value} in {round((finish-start)*1000)} ms')
    return value

# All products of triangle components (v1x*v1y, v1x*v1z, ..., v1x*v3z, etc.)
P: Dict[str, List[float]] = {}
for i in range(0, 3):
    for j in range(0, 3):
        for k in range(0, 3):
            for l in range(0, 3):
                P[f'{i+1}{chr(ord("x")+j)},{k+1}{chr(ord("x")+l)}'] = \
                    np.float128(original_mesh.vectors[:, i, j]) * \
                    np.float128(original_mesh.vectors[:, k, l])

sum1z2x1x2z1z3x2z3x1x3z2x3z = P['1z,2x'] - P['1x,2z'] - P['1z,3x'] + P['2z,3x'] + P['1x,3z'] - P['2x,3z']
sum1y2x1x2y1y3x2y3x1x3y2x3y = P['1y,2x'] - P['1x,2y'] - P['1y,3x'] + P['2y,3x'] + P['1x,3y'] - P['2x,3y']
sum1y2z2z3y1z2y1z3y1y3z2y3z = P['1y,2z'] - P['2z,3y'] - P['1z,2y'] + P['1z,3y'] - P['1y,3z'] + P['2y,3z']
sum1z2y1y2z1z3y2z3y1y3z2y3z = P['1z,2y'] - P['1y,2z'] - P['1z,3y'] + P['2z,3y'] + P['1y,3z'] - P['2y,3z']
def dfdt(t) -> List[float]:
    start = time.time()
    cosx = math.cos(t[0])
    sinx = math.sin(t[0])
    cosy = math.cos(t[1])
    siny = math.sin(t[1])

    cosx_component =     sum1z2x1x2z1z3x2z3x1x3z2x3z*cosx
    sinx_component =    -sum1y2x1x2y1y3x2y3x1x3y2x3y*sinx
    cosy_component =     sum1y2z2z3y1z2y1z3y1y3z2y3z*cosy
    siny_component =     sum1z2y1y2z1z3y2z3y1y3z2y3z*siny
    cosxcosy_component = sum1y2x1x2y1y3x2y3x1x3y2x3y*(cosx*cosy)
    sinxsiny_component = sum1z2x1x2z1z3x2z3x1x3z2x3z*(sinx*siny)
    cosysinx_component = sum1z2x1x2z1z3x2z3x1x3z2x3z*(cosy*sinx)
    cosxsiny_component = sum1y2x1x2y1y3x2y3x1x3y2x3y*(cosx*siny)

    xpart2 = cosx_component + sinx_component
    xpart3 = cosxcosy_component + cosysinx_component + siny_component

    ypart2 = xpart3
    ypart3 = cosy_component + cosxsiny_component + sinxsiny_component
    
    divisor = 2*np.abs(xpart3)
    sign = np.sign(xpart3)

    def x():
        acc = np.divide(cosy * sign * xpart2 * xpart3, divisor, where=(divisor != 0))
        return np.sum(acc)

    def y():
        acc = -np.divide(sign * ypart2 * ypart3, divisor, where=(divisor != 0))
        return np.sum(acc)

    dt = np.array([x(), y(), 0])

    finish = time.time()
    print(f'Computed dfdt({t})={dt} in {round((finish-start)*1000)}ms')
    return dt


def dfdty(t) -> float:
    pass


res = optimize.dual_annealing(
  f, bounds=[[-math.pi, math.pi], [-math.pi, math.pi]])
# res = optimize.minimize(fun=f, jac=None, method='BFGS', bounds=[(0, math.pi*2), (0, math.pi*2), (0, 0)], x0=np.array([0, 0, 0]), options={'xatol': 1E-10})
print(f'f({np.round(res.x, decimals=2)})={f(res.x)}')

def plot_f():
    resolution = [20,20]

    x_rotation = np.linspace(-math.pi, math.pi, resolution[0])
    y_rotation = np.linspace(-math.pi, math.pi, resolution[1])
    x_rotation_mesh, y_rotation_mesh = np.meshgrid(
        x_rotation, y_rotation, indexing='ij')


    f_of_t = np.array([[f([x, y, 0]) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(x_rotation_mesh, y_rotation_mesh)])
    s = ax.plot_surface(x_rotation_mesh, y_rotation_mesh, f_of_t)

    # dfdtx_of_t = np.array([[dfdt([x, y, 0])[0] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(x_rotation_mesh, y_rotation_mesh)])
    # s = ax.plot_surface(x_rotation_mesh, y_rotation_mesh, dfdtx_of_t)
    # dfdty_of_t = np.array([[dfdt([x, y, 0])[1] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(x_rotation_mesh, y_rotation_mesh)])
    # s = ax.plot_surface(x_rotation_mesh, y_rotation_mesh, dfdty_of_t)

    #ax.scatter(res.x[0], res.x[1], f(res.x), c='red')
    #ax.scatter(math.pi/2, 0, f([math.pi,0]), c='green')
    plt.show()

def plot_stl(x):
    original_mesh.rotate([1, 0, 0], x[0])
    original_mesh.rotate([0, 1, 0], x[1])
    if len(x) > 2:
        original_mesh.rotate([0, 0, 1], x[2])
    stl_polygons = mplot3d.art3d.Poly3DCollection(original_mesh.vectors)
    stl_polygons.set_facecolor('gold')
    stl_polygons.set_edgecolor('black')
    ax.add_collection3d(stl_polygons)
    scale = original_mesh.points.ravel()
    ax.auto_scale_xyz(scale, scale, scale)
    plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Effect of Rotating {FILE_NAME} on Function Value')
ax.set_xlabel('X-rotation')
ax.set_ylabel('Y-rotation')
ax.set_zlabel('Function value')

#plot_f()
plot_stl([math.pi,0])
# plt.colorbar(s)
#f([0,0,0])
#f([math.pi/2,0,0])
#f([math.pi,0,0])
