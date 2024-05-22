#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 16:26:05 2022

@author: rlg3


Edge Dislocation Functions

New functions created on May 5 2023

@author: yfwang09

Displacement gradient functions

Fg_test: pure shear in a sphere
Fg_disl: single edge dislocation
Fg_load: load a displacement gradient field from file
"""

import os
from math import pi, sqrt
import numpy as np
import displacement_grad_helper as dgh

def return_dis_grain_matrices_all():
    dis_grain_all = np.zeros([3,3,12])
    dis_grain_all[:,:,0] = [[1/sqrt(2), 1/sqrt(2), 0], [-1/sqrt(3), 1/sqrt(3), 1/sqrt(3)],\
                            [1/sqrt(6), -1/sqrt(6), 2/sqrt(6) ]]
    dis_grain_all[:,:,1] = [[1/sqrt(2), 1/sqrt(2), 0], [1/sqrt(3), -1/sqrt(3), 1/sqrt(3)],\
                            [1/sqrt(6), -1/sqrt(6), -2/sqrt(6) ]]
    dis_grain_all[:,:,2] = [[1/sqrt(2), -1/sqrt(2), 0], [-1/sqrt(3), -1/sqrt(3), -1/sqrt(3)],\
                            [1/sqrt(6), 1/sqrt(6), -2/sqrt(6) ]]
    dis_grain_all[:,:,3] = [[1/sqrt(2), -1/sqrt(2), 0], [1/sqrt(3), 1/sqrt(3), -1/sqrt(3)],\
                            [1/sqrt(6), 1/sqrt(6), 2/sqrt(6) ]]
    dis_grain_all[:,:,4] = [[1/sqrt(2), 0, -1/sqrt(2)], [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)],\
                            [1/sqrt(6), -2/sqrt(6), 1/sqrt(6) ]]
    dis_grain_all[:,:,5] = [[1/sqrt(2), 0, -1/sqrt(2)], [1/sqrt(3), -1/sqrt(3), 1/sqrt(3)],\
                            [-1/sqrt(6), -2/sqrt(6), -1/sqrt(6) ]]
    dis_grain_all[:,:,6] = [[1/sqrt(2), 0, 1/sqrt(2)], [1/sqrt(3), 1/sqrt(3), -1/sqrt(3)],\
                            [-1/sqrt(6), 2/sqrt(6), 1/sqrt(6) ]]
    dis_grain_all[:,:,7] = [[1/sqrt(2), 0, 1/sqrt(2)], [-1/sqrt(3), 1/sqrt(3), 1/sqrt(3)],\
                            [-1/sqrt(6), -2/sqrt(6), 1/sqrt(6) ]]
    dis_grain_all[:,:,8] = [[0, 1/sqrt(2), 1/sqrt(2)], [1/sqrt(3), 1/sqrt(3), -1/sqrt(3)],\
                            [-2/sqrt(6), 1/sqrt(6), -1/sqrt(6) ]]
    dis_grain_all[:,:,9] = [[0, 1/sqrt(2), 1/sqrt(2)], [-1/sqrt(3), 1/sqrt(3), -1/sqrt(3)],\
                             [-2/sqrt(6), -1/sqrt(6), 1/sqrt(6) ]]
    dis_grain_all[:,:,10] = [[0, 1/sqrt(2), -1/sqrt(2)], [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)],\
                             [2/sqrt(6), -1/sqrt(6), -1/sqrt(6) ]]
    dis_grain_all[:,:,11] = [[0, 1/sqrt(2), -1/sqrt(2)], [-1/sqrt(3), 1/sqrt(3), 1/sqrt(3)],\
                             [2/sqrt(6), 1/sqrt(6), 1/sqrt(6) ]]
    return dis_grain_all
    
def return_dis_grain_matrices(b=None, n=None, t=None):
    ''' Returns the rotation matrix from the dislocation coordinates to the sample (Miller) coordinates.
     
    This function determines the rotation matrix from the dislocation coordinates to the sample coordinates (Miller indices). The sample coordinates always define the y||n and z||t, the x is defined based on y and z, since the Burger's vector could be in any direction.
    
    The vectors don't have to be normalized, but the orthogonality will be checked, and the function also makes sure both the Burger's vector and the line direction vector are both on the slip plane.

    Important Notes:
    * n and t must be given to determine the coordinate system
    * Otherwise, the function will return all the slip systems as a (3,3,12) array.

    Parameters
    ----------
    b : array of length 3, by default None
        Burger's vector in Miller, doesn't have to be normalized
    n : array of length 3, by default None
        normal vector of the slip plane, doesn't have to be normalized
    t : array of length 3, by default None
        line direction of the dislocation, doesn't have to be normalized

    Returns
    -------
    Ud : array of shape (3,3) or (3,3,12)
        rotation matrix that converts dislocation coordinates into the sample coordinates 
    '''

    if (n is None) or (t is None):
        return return_dis_grain_matrices_all()
    if np.dot(n, t) != 0:
        raise ValueError('return_dis_grain_matrices: t must be on the plane n')
    yd = n/np.linalg.norm(n)
    zd = t/np.linalg.norm(t)
    xd = np.cross(yd, zd)
    Ud = np.transpose([xd, yd, zd])
    return Ud

def Fg_test(d, xg, yg, zg, R0=1.5e-6, components=(2,0), strain=5e-4):
    ''' Returns the F-tensor field for a pure shear strain field within a sphere.

    Parameters
    ----------
    d : dict
        resolution function input dictionary (not used)
    xg : float, or array
        x coordinate in the grain system
    yg :float, or array (shape must match with xg)
        y coordinate in the grain system
    zg :float, or array (shape must match with xg)
        z coordinate in the grain system
    R0 : float, by default 1.5e-6
        radius of the sphere (in m)
    components : tuple of length 2, by default (2,0)
        the components of the strain tensor to apply the strain
    strain : float, by default 5e-4
        the strain value to apply

    Returns
    -------
    Fd : numpy array
        shape(xg)x3x3 strain tensor
    '''

    Fg = np.zeros(xg.shape + (3, 3))
    r = np.sqrt(xg**2 + yg**2 + zg**2)
    rind = r < R0
    i, j = components
    Fg[rind, i, j] = strain
    Fg += np.eye(3) # add the identity tensor to convert to the deformation tensor

    return Fg

def Fg_disl(d, xg, yg, zg):
    ''' Returns the dislocation strain tensor in the sample coordinates.
    
    The dislocation strain tensor is calculated in the dislocation coordinates, and then rotated into the sample coordinates.

    Parameters
    ----------
    d : dict
        resolution function input dictionary
    xg : float, or array
        x displacement from dislocation core
    yg :float, or array (shape must match with xg)
        y displacement from dislocation core
    zg :float, or array (shape must match with xg)
        z displacement from dislocation core

    Returns
    -------
    Fd : numpy array
        shape(xg)x3x3 strain tensor
    '''
    
    # get grain rotation matrix
    if 'Ug' in d.keys():
        Ug = d['Ug']
    else:
        Ug = np.eye(3)
    if ('hkl' in d.keys()) and ('xlab' in d.keys()) and ('ylab' in d.keys()): # for compatibility with old code
        Ug = return_dis_grain_matrices(b=d['xlab'], n=d['ylab'], t=d['hkl']).T
    if ('hkl' in d.keys()) and ('xcry' in d.keys()) and ('ycry' in d.keys()):
        Ug = return_dis_grain_matrices(b=d['xcry'], n=d['ycry'], t=d['hkl']).T
        # print(Ug)
        # Ug = np.linalg.inv(Ug)
        # print(Ug)
    # get dislocation coordinates
    Ud = return_dis_grain_matrices(b=d['bs'], n=d['ns'], t=d['ts']) # shape (3,3)
    rg = np.stack([xg, yg, zg], axis=-1) # shape (xg.shape, 3)
    rc = np.einsum('ij,...j->...i', Ug, rg) # shape (xg.shape, 3)
    rd = np.einsum('ij,...j->...i', Ud.T, rc) # shape (xg.shape, 3)
    xd, yd = rd[...,0], rd[...,1] # shape (xg.shape)
    # get strain tensor in the dislocation coordinates
    Fd = get_disl_strain_tensor(d, xd, yd)
    # rotate into the crystal coordinates (Miller indices)
    Fc = np.einsum('ij,...jk,kl->...il', Ud, Fd, Ud.T)
    # rotate into the grain coordinates
    Fg = np.einsum('ij,...jk,kl->...il', Ug, Fc, Ug.T)
    
    return Fg

def Fg_disl_network(d, xg, yg, zg, filename=None):
    ''' Returns the dislocation strain tensor in the sample coordinates.

    Use the non-singular displacement gradient function

    Parameters
    ----------
    d : dict
        resolution function input dictionary
    xg : float, or array
        x coordinate in the grain system
    yg :float, or array (shape must match with xg)
        y coordinate in the grain system
    zg :float, or array (shape must match with xg)
        z coordinate in the grain system

    Returns
    -------
    Fg : numpy array
        shape(xg)x3x3 strain tensor
    '''
    # get grain rotation matrix
    if 'Ug' in d.keys():
        Ug = d['Ug']
    else:
        Ug = np.eye(3)
    if ('hkl' in d.keys()) and ('xcry' in d.keys()) and ('ycry' in d.keys()):
        Ug = return_dis_grain_matrices(b=d['xcry'], n=d['ycry'], t=d['hkl']).T
    if 'nu' in d.keys():
        NU = d['nu']
    else:
        NU = 0.324
    if 'a' in d.keys():
        a = d['a']
    else:
        a = 1.0
    if 'rn' in d.keys():
        rn = d['rn']
    else:
        rn = np.array([[ 78.12212123, 884.74707189, 483.30385117],
                       [902.71333272, 568.95913492, 938.59105117],
                       [500.52731411, 261.22281654, 552.66098404]])
    if 'links' in d.keys():
        links = d['links']
    else:
        links = np.transpose([[0, 1, 2], [1, 2, 0]])
    if 'bs' in d.keys():
        b = d['bs']
        b = b/np.linalg.norm(b)
        # if 'b' in d.keys():
        #     b = b*d['b']
    else:
        b = np.array([1, 1, 0])
        b = b/np.linalg.norm(b)
    if 'b' in d.keys():
        bmag = d['b']
    else:
        bmag = 1
    if 'ns' in d.keys():
        n = d['ns']
    else:
        n = np.array([1, 1, 1])
    n = n/np.linalg.norm(n)

    if links.shape[1] == 2: # only connectivity is provided
        links = np.hstack([links, np.tile(b, (3, 1)), np.tile(n, (3, 1))])
    elif links.shape[1] != 8:
        raise ValueError('links array must include b and n')
    r_obs = np.stack([xg.flatten(), yg.flatten(), zg.flatten()], axis=-1)
    rnorm = r_obs/bmag
    rn = rn/bmag

    if filename is not None and os.path.exists(filename):
        Fg_list = np.load(filename)['Fg']
    else:
        test = (filename == 'test')
        Fg_list = dgh.displacement_gradient_structure_matlab(rn, links, NU, a, rnorm, test=test)

    if filename is not None:
        np.savez(filename, Fg=Fg_list, r_obs=r_obs)
    Fg = np.reshape(Fg_list, xg.shape + (3, 3)) + np.identity(3)
    
    return Fg

def get_disl_strain_tensor(d, xd, yd):
    '''
    Returns dislocation strain tensor
    (currently just for edge-type dislocation)

    z: dislocation line direction
    y: normal of the slip plane
    x: Burger's vector direction (edge dislocation)

    Parameters
    ----------
    d : dict
        resolution function input dictionary
    xd : float, or array
        x displacement from dislocation core
    yd :float, or array (shape must match with xd)
        y displacement from dislocation core

    Returns
    -------
    Fd : numpy array
        shape(xd)x3x3 strain tensor
    '''
    prefactor = d['b']/(4*pi*(1-d['nu']))
    A = 2 * d['nu'] * (xd**2 + yd**2)       # xd.shape()
    denom = (xd**2 + yd**2)**2              # xd.shape()

    # All the following becomes xd.shape()x1x1
    Fxx = (-prefactor * yd * (3 * xd**2 + yd**2 - A) / denom)[..., None, None]
    Fxy = (prefactor * xd * (3 * xd**2 + yd**2 - A) / denom)[..., None, None]
    Fyx = (-prefactor * xd * (xd**2 + 3 * yd**2 - A) / denom)[..., None, None]
    Fyy = (prefactor * yd * (xd**2 - yd**2 - A) / denom)[..., None, None]

    O = np.zeros_like(Fxx) # zero-filler

    Fd = np.block([[Fxx, Fxy, O], [Fyx, Fyy, O], [O, O, O]]) + np.identity(3)
    return Fd

def get_disl_strain_tensor_single(d, xd, yd):
    '''
    Returns dislocation strain tensor
    (currently just for edge-type dislocation)

    Parameters
    ----------
    d : dict
        resolution function input dictionary
    xd : float
        x displacement from dislocation core
    yd :float
        y displacement from dislocation core

    Returns
    -------
    Fd : numpy array
        3x3 strain tensor
    '''
    prefactor = d['b']/(4*pi*(1-d['nu']))
    A = 2 * d['nu'] * (xd**2 + yd**2)
    denom = (xd**2 + yd**2)**2

    Fxx = float(-prefactor * yd * (3 * xd**2 + yd**2 - A) / denom)
    Fxy = float(prefactor * xd * (3 * xd**2 + yd**2 - A) / denom)
    Fyx = float(-prefactor * xd * (xd**2 + 3 * yd**2 - A) / denom)
    Fyy = float(prefactor * yd * (xd**2 - yd**2 - A) / denom)

    Fd = np.array([[Fxx, Fxy, 0], [Fyx, Fyy, 0], [0, 0, 0]]) + np.identity(3)
    return Fd




