from typing import Dict, List, Callable
import math
import time

import numpy as np
import stl

# All used products of triangle components (v1x*v1y, v1x*v1z, ..., v1x*v3z, etc.)
# see my Mathematica notebook for where these come from
def compute_sums_and_products(mesh: stl.mesh.Mesh) -> Dict[str, List[float]]:
    sp: Dict[str, List[float]] = {}
    i: int
    j: int
    k: int
    l: int
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    product_string: str = f'{i+1}{chr(ord("x")+j)}{k+1}{chr(ord("x")+l)}'
                    product_string_rev: str = f'{k+1}{chr(ord("x")+l)}{i+1}{chr(ord("x")+j)}'
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
    sp: Dict[str, List[float]] = compute_sums_and_products(mesh)
    def f(theta: List[float]) -> float:
        if debug:
            start: float = time.time()

        sinx: float = math.sin(theta[0])
        cosx: float = math.cos(theta[0])
        siny: float = math.sin(theta[1])
        cosy: float = math.cos(theta[1])
        cosxcosy: float = cosx*cosy
        cosysinx: float = cosy*sinx

        S: List[float] = sp['sa']*cosxcosy + sp['sb']*cosysinx + sp['sc']*siny

        z_height: List[float] = mesh.vectors[:,:,2]*cosxcosy - mesh.vectors[:,:,1]*cosysinx + mesh.vectors[:,:,0]*siny
        z_min: float = np.min(z_height)

        overhang_bias: List[float] = (1 + np.tanh(S))

        bottom_face_bias: List[float] = np.tanh((np.sum(z_height ,axis=1) - 3*z_min)/10)


        value: float = .25 * np.sum(np.abs(S) * overhang_bias * bottom_face_bias)

        if debug:
            finish: float = time.time()
            print(f'Computed f({theta})={value} in {round((finish-start)*1000)} ms')

        return value
    return f
