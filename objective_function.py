from typing import Dict, List, Callable
import math
import time

import numpy as np
import stl

# All used products of triangle components (v1x*v1y, v1x*v1z, ..., v1x*v3z, etc.)
# see my Mathematica notebook for where these come from
def compute_sums_and_products(mesh: stl.mesh.Mesh) -> Dict[str, List[float]]:
    sp: Dict[str, List[float]] = {}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    product_string = f'{i+1}{chr(ord("x")+j)}{k+1}{chr(ord("x")+l)}'
                    product_string_rev = f'{k+1}{chr(ord("x")+l)}{i+1}{chr(ord("x")+j)}'
                    if i == k or j == l: # Squares (1x1x) are never used, neither are 1x2x or 2x2y forms
                        continue
                    if product_string_rev in sp: # When 2x1y exists, set 1y2x equal to it, instead of redoing
                        sp[product_string] = sp[product_string_rev]
                        continue
                    sp[f'{i+1}{chr(ord("x")+j)}{k+1}{chr(ord("x")+l)}'] = \
                        mesh.vectors[:, i, j] * \
                        mesh.vectors[:, k, l]
    sp['sa'] = sp['1y2x'] - sp['1x2y'] - sp['1y3x'] + sp['2y3x'] + sp['1x3y'] - sp['2x3y']
    sp['sb'] = sp['1z2x'] - sp['1x2z'] - sp['1z3x'] + sp['2z3x'] + sp['1x3z'] - sp['2x3z']
    sp['sc'] = sp['1z2y'] - sp['1y2z'] - sp['1z3y'] + sp['2z3y'] + sp['1y3z'] - sp['2y3z']
    
    return sp



def build_f(mesh: stl.mesh.Mesh, debug: bool) -> Callable[[List[float]], float]:
    sp = compute_sums_and_products(mesh)
    def f(theta: List[float]) -> float:
        if debug:
            start = time.time()

        sinx = math.sin(theta[0])
        cosx = math.cos(theta[0])
        siny = math.sin(theta[1])
        cosy = math.cos(theta[1])
        cosxcosy = cosx*cosy
        cosysinx = cosy*sinx

        S = sp['sa']*cosxcosy + sp['sb']*cosysinx + sp['sc']*siny

        z_height = mesh.vectors[:,:,2]*cosxcosy - mesh.vectors[:,:,1]*cosysinx + mesh.vectors[:,:,0]*siny
        z_min = np.min(z_height)

        # Add a bias to emphasize overhang surfaces.
        # 
        # The function computes the total area of the 2D projections of the part. To minimize overhang area in particular,
        # there are several possible treatments:
        # 1. Set overhang area to be positive and top surface area to be negative. However, for some 
        #    symmetric cases like a cube, this will always be 0.
        # 2. Set overhang area to be positive and top surface area to be 0. This is ideal using the unit step function
        #    but results in discontinuous derivatives.
        # 
        # I use a variation on #2: a unit-step approximation. This perturbs the function appropriately and
        # keeps all partial derivatives continuous.
        overhang_bias = (1 + np.tanh(S))        

        # Add a bias so that a flat bottom face parallel to the build plate is not treated as an overhang.
        # 
        # A triangle with all three points at the lowest vertex height is parallel to the build plate. Since it
        # is also the lowest face of the part, it lies flat on the build plate and is not an overhang. The function
        # should take this into account.
        #
        # 1. Add an if statement to set the area to 0 if a triangle's vertices are all at the minimum z-height. Again,
        #    like with the overhang bias, this results in discontinuous derivatives. For a non-derivative method, the 
        #    function would have no gradient to suggest that it is better to orient a part with big, flat surfaces on 
        #    the bottom.
        #
        # I use a variation on #1: the hyperbolic tangent of the sum of the z-coordinates' distance from z-min.
        # As all three coordinates near z-min, the value approaches 0.
        # The constant used here requires some tweaking; larger constants improve the rate of convergence
        # to the global minimizer, but can also prune global minimizers for shapes with many faces that are
        # near-bottom faces. Consider the reuleaux tetrahedron example, which is positioned on a side rather 
        # than on a point, which should be the case without this bias.
        bottom_face_bias = np.tanh((np.sum(z_height ,axis=1) - 3*z_min)/10)


        value = .25 * np.sum(np.abs(S) * overhang_bias * bottom_face_bias)

        if debug:
            finish = time.time()
            print(f'Computed f({theta})={value} in {round((finish-start)*1000)} ms')

        return value
    return f
