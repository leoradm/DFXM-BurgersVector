#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 10:43:19 2022

@author: rlg3


Forward Model Set Up. Adaption from Julia code.
Next: write include screw dislocation characters.


Citations:
    1. https://arxiv.org/pdf/2007.09475.pdf (Forward model paper)
    2. https://arxiv.org/ftp/arxiv/papers/2009/2009.05083.pdf (Al disl boundary)
    
Questions:
    1. voel --> detector transformation
"""

from math import pi, cos, sin, sqrt, tan
import time
import numpy as np
from numpy.random import rand
from scipy.stats import gaussian_kde, truncnorm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import edge_disl_helper as edge

'''
Instrument Parameters for Resolution function
'''
res_dict = {
    'zeta_v_rms' : 0.03*(pi/180)/2.355, #[2-sup, pg 2] incoming divergence in vertical direction in rad (about 0.523 mrad)
    'zeta_h_rms': 1E-5/2.355, # [Henning's code] incoming divergence in horizontal direction, in rad 
    'NA_rms' : 7.2E-4, # [2-sup, pg 2] NA of objective, in rad 
    'eps_rms' : 1.4E-4/2.355, #[1, pg ] rms width of x-ray energy bandwidth (2.355 factor fwhm -> rms)
    'zl_rms' : 600/2.355, # rms width of the Gaussian beam profile
    'two_theta' : 20.73, # [2-sup, pg 2] 2θ in degrees
    'D' : 435, # [2-sup, pg 2] physical aperture of objective, in m
    'd1' : 0.274, # [2-sup, pg 2] sample-objective distance, in m
    'TwoDeltaTheta' : 0, #rotation of the 2 theta arm
    'phi' : 0, #(1/6)*(pi/180), #in rad; sample tilt angle 1
    'chi' : 0.015*pi/180, #in rad, sample tilt angle 2
    'hkl' : [0, 0, 1],
    'b' : 0.286,
    'nu' : .334
    }

res_dict['mu'] = res_dict['two_theta']/2*(pi/180) #in rad; base tilt
res_dict["theta"] = res_dict['two_theta']/2*(pi/180)


'''
Definition of dislocation characters

Matrices encode the grain --> dislocation transformation
'''


'''
Constant Transformation Matrices
'''
Omega = np.identity(3)
sample_grain = np.identity(3)
    
def get_rot_matrices(d : dict):
    '''
    Returns rotation matrices reuired to go from sample to lab frame

    Parameters
    ----------
    d : dict
        resolution function input dictionary

    Returns
    -------
    Chi : numpy array
        DESCRIPTION.
    Phi : numpy array
        DESCRIPTION.
    Mu : numpy array
        DESCRIPTION.

    '''
    Chi = np.array([[1, 0, 0], [0, cos(d['chi']), -sin(d['chi'])],\
                    [0, sin(d['chi']), cos(d['chi'])]])
    Phi = np.array([[cos(d['phi']), 0, -sin(d['phi'])], [0, 1, 0],\
                    [sin(d['phi']), 0, cos(d['phi'])]])
    Mu = np.array([[cos(d['mu']), 0, sin(d['mu'])], [0, 1, 0],\
                       [-sin(d['mu']), 0, cos(d['mu'])]])
    return Chi, Phi, Mu

def get_image_lab(d):
    '''
    Transformation matrix for lab --> image reference frames

    Parameters
    ----------
    d : dict
        resolution function input dictionary

    '''
    theta = d['two_theta']/2*(pi/180)
    theta_p = theta + d['TwoDeltaTheta']

    Theta = np.array([[cos(theta_p), 0, sin(theta_p)],\
             [0, 1, 0], [-sin(theta_p), 0, cos(theta_p)]])
    Theta2 = Theta*Theta
    
    image_lab = Theta2
    return image_lab    
    
    

def get_bnt(d : dict, dis_grain_all, char):
    '''
    Returns three vectors specifying dislocation: Burgers vector, slip plane
    normal, line vector

    Parameters
    ----------
    phi : float
        Sample tilt angle 1
    chi : float
        DESCRIPTION.
    TwoDeltaTheta : float
        DESCRIPTION.
    char : integer
        Character of dislocation

    Returns
    -------
    [Burgers vector, line vector, slip-plane normal] in 3 reference frames:
        1. Sample frame
        2. Image frame
        3. Lab frame

    '''
    image_lab = get_image_lab(d)
    Chi, Phi, Mu = get_rot_matrices(d)
    Omega = np.identity(3)
    lab_sample = np.matmul(np.matmul(Mu,Omega), np.matmul(Chi,Phi))
    image_sample = np.matmul(image_lab, lab_sample)
    
    dis_grain = dis_grain_all[:,:,char]
    
    # burgers, line, slip plane normal vectors
    bnt_sample = np.transpose(dis_grain)
    bnt_im = np.matmul(image_sample, bnt_sample)
    bnt_lab = np.matmul(lab_sample, bnt_sample)
    
    return bnt_sample, bnt_im, bnt_lab

def get_dis_lab(d : dict, dis_grain_all, char):
    '''
    Transformation matrix for lab --> dislocation reference frames

    Parameters
    ----------
    d : dict
        resolution function input dictionary
    char : int
        character of the dislocation

    '''
    Chi, Phi, Mu = get_rot_matrices(d)
    Omega = np.identity(3)
    lab_sample = np.matmul(np.matmul(Mu,Omega), np.matmul(Chi,Phi))
    grain_lab = np.transpose(np.matmul(lab_sample, sample_grain))

    # define coordinate transforms that depend on dislocation character
    dis_grain = dis_grain_all[:,:,char]

    dis_lab = np.matmul(dis_grain, grain_lab)

    return dis_lab

def define_voxel(d, Npixels, psize, zsize):
    '''
    Returns 1D arrays specifying voxel locations. Assumed same for x, y, and z
    locations (cubic voxels)

    Parameters
    ----------
    Npixels : integer
        Number of pixels across the detector. Same in both y and z.
    psize : float
        Pixel size in nm.
        
    zsize : float
        ???

    Returns
    -------
    x_lab_vec, y_lab_vec, z_lab_vec

    '''
    theta = d['two_theta']/2*(pi/180)
    #zbound = 3*zsize/2
    zbound = (Npixels-1)*psize/2
    zlab_vec = np.linspace(-zbound, zbound, Npixels)
    ybound = (Npixels-1)*psize/2
    ylab_vec = np.linspace(-ybound, ybound, Npixels)
    xbound = psize/(2*tan(2*theta))
    xlab_vec = np.linspace(-xbound, xbound, Npixels)
    return xlab_vec, ylab_vec, zlab_vec

    

def res_fxn_q(d):
    '''
    Intensity functions for qrock_prime, qroll, and qpar as a function of q-vector

    Parameters
    ----------
    d : TYPE
        DESCRIPTION.

    Returns
    -------
    q_fxn : TYPE
        DESCRIPTION.

    '''
    #Monte Carlo simulation to sample delta_2theta and xi
    theta = d['two_theta']/2*(pi/180)
    Nrays = int(1e4)
    delta_2theta = np.zeros(Nrays)
    xi = np.zeros(Nrays)
    phys_aper = d['D']/d['d1']
    for k in range(Nrays):    
        temp = phys_aper/2 + 1
        while abs(temp) > phys_aper/2:
            temp = rand()*d['NA_rms']
        delta_2theta[k] = temp
    
        temp = phys_aper/2 + 1
        while abs(temp) > phys_aper/2:
            temp = rand()*d['NA_rms']
        xi[k] = temp
    #Compute zeta_v and zeta_h : divergence of synchotron source or condenser optics
    zeta_v = (rand(Nrays)-0.5)*d['zeta_v_rms']*2.355
    zeta_h =rand(Nrays)*d['zeta_v_rms']
    eps = rand(Nrays) * d['eps_rms']

    
    # compute q_{rock,roll,par} 
    qrock = -zeta_v/2 - delta_2theta/2 + d['phi']
    qroll = -zeta_h/(2*sin(theta)) -  xi/(2*sin(theta))
    qpar = eps + (1/tan(theta))*(-zeta_v/2 + delta_2theta/2)
    qrock_prime = cos(theta)*qrock + sin(theta)*qpar
    
    
    q_fxn = {'Qrock_prime' : {'q' : qrock_prime, 'intensity_fxn' : []},\
             'QParallel' : {'q' : qpar, 'intensity_fxn' : []},\
                 'QRoll' : {'q' : qroll, 'intensity_fxn' : []}}
        
    # Kerney Density Estimator. bw_method: bandwidth estimation method
    for k,v in q_fxn.items():
        res_max = max(gaussian_kde(v['q'], bw_method='silverman').pdf(v['q']))
        #res_max = max(gaussian_kde(v['q'], bw_method='silverman').pdf(np.arange(-5e-3, 1e-1, 1e-6)))
        v['intensity_fxn'] = lambda q: gaussian_kde(v['q'],\
                                                    bw_method='silverman').pdf(q) / res_max
    return q_fxn

def plot_res(qrock, qroll, qpar, plot_half_range=4.5e-3, show=True):
    Nrays = len(qrock)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    Nskip = Nrays//10000 + 1
    
    qrock = qrock[::Nskip]
    qroll = qroll[::Nskip]
    qpar = qpar[::Nskip]
    ax.plot(qrock, qroll, qpar, 'ok', markersize=1)
    ax.plot(np.zeros_like(qrock)-plot_half_range, qroll, qpar, 'o', markersize=1)
    ax.plot(qrock, np.zeros_like(qroll)+plot_half_range, qpar, 'o', markersize=1)
    ax.plot(qrock, qroll, np.zeros_like(qpar)-plot_half_range, 'o', markersize=1)
    ax.set_xlim([-plot_half_range, plot_half_range])
    ax.set_ylim([-plot_half_range, plot_half_range])
    ax.set_zlim([-plot_half_range, plot_half_range])
    if show:
        plt.show()
    else:
        return fig, ax

def res_fn(d, Nrays=10000, q1_range=1e-3, q2_range=5e-3, q3_range=5e-3, npoints1=40, npoints2=40, npoints3=40, plot=False, saved_q=None, timeit=False):
    ''' Compute the resolution function for DFXM
        The objective is modelled as an isotropic Gaussian with an NA and in addition a square phyical aperture of d side length D. 
        
        Yifan Wang, Mar 14, 2023, version 1.2

    Parameters
    ----------
    d : dict
        dictionary for dislocation and instrumental settings
    Nrays : int, default 10000
        number of rays to simulate the resolution function
    q1_range : float, default 1e-3
        range of q1 to compute the resolution function
    q2_range : float, default 5e-3
        range of q2 to compute the resolution function
    q3_range : float, default 5e-3
        range of q3 to compute the resolution function
    npoints1 : int, default 40
        number of points in q1 to compute the resolution function
    npoints2 : int, default 40
        number of points in q2 to compute the resolution function
    npoints3 : int, default 40
        number of points in q3 to compute the resolution function
    plot : bool, default False
        whether to plot the resolution function
    saved_q : tuple, default None
        if not None, the qvectors will be loaded from saved_q

    Returns
    -------
    Res_qi : array of (Npixels, Npixels, Npixels)
        3D voxelized field of resolution function
    ratio_outside : float
        ratio of rays outside the physical aperture
    '''

    # INPUT instrumental settings
    if timeit:
        tic = time.time()

    if 'q1_range' in d.keys():
        q1_range = d['q1_range']
    if 'q2_range' in d.keys():
        q2_range = d['q2_range']
    if 'q3_range' in d.keys():
        q3_range = d['q3_range']
    if 'npoints1' in d.keys():
        npoints1 = d['npoints1']
    if 'npoints2' in d.keys():
        npoints2 = d['npoints2']
    if 'npoints3' in d.keys():
        npoints3 = d['npoints3']
    if 'NA_rms' not in d.keys():
        d['NA_rms'] = 7.31e-4/2.35      # NA of objective, in rad (why divide by 2.35?)
    if 'zeta_v_rms' not in d.keys():
        d['zeta_v_rms'] = 0.53e-3/2.35  # incoming divergence in vertical direction, in rad
    if 'zeta_h_rms' not in d.keys():
        d['zeta_h_rms'] = 1e-5/2.35     # incoming divergence in horizontal direction, in rad
    if 'eps_rms' not in d.keys():
        d['eps_rms'] = 0.00006          # rms width of x-ray energy bandwidth
    if 'D' not in d.keys():
        d['D'] = 2*np.sqrt(50e-6*1e-3)  # physical aperture of objective, in m
    if 'd1' not in d.keys():
        d['d1'] = 0.274                 # sample-objective distance, in m
    if 'phi' not in d.keys():
        d['phi'] = 0                    # sample rocking angle, in rad
    if 'two_theta' not in d.keys():
        d['two_theta'] = 20.73          # scattering angle for fcc (001), in deg
    d['theta'] = np.deg2rad(d['two_theta']/2)        # half scattering angle, in rad
    phys_aper = d['D']/d['d1']          # physical aperture of objective, in rad

    ######## Ray tracing in crystal system ########
    # Eq. 43-45 in the Poulsen et al. (2021) paper

    # Sample incoming rays
    if saved_q is None:
        # zeta_v = np.random.randn(Nrays)*d['zeta_v_rms']
        zeta_v = (np.random.rand(Nrays) - 0.5)*d['zeta_v_rms']*2.35 # using uniform distribution to be consistent with Henning's implementation
        zeta_h = np.random.randn(Nrays)*d['zeta_h_rms']
        eps    = np.random.randn(Nrays)*d['eps_rms']

        # Define truncated normal distribution by the physical aperture
        delta_2theta = truncnorm.rvs(-phys_aper/2/d['NA_rms'], phys_aper/2/d['NA_rms'], size=Nrays) * d['NA_rms']
        xi = truncnorm.rvs(-phys_aper/2/d['NA_rms'], phys_aper/2/d['NA_rms'], size=Nrays) * d['NA_rms']

        if timeit:
            print('Time for sampling rays: {:.2f} s'.format(time.time()-tic))

        # Compute q_{rock,roll,par}
        qrock = -zeta_v/2 - delta_2theta/2 # + d['phi']
        qroll = -zeta_h/(2*np.sin(d['theta'])) - xi/(2*np.sin(d['theta'])) # + d['chi']
        qpar = eps + (1/np.tan(d['theta']))*(-zeta_v/2 + delta_2theta/2)

        # Convert from crystal to imaging system
        qrock_prime = np.cos(d['theta'])*qrock + np.sin(d['theta'])*qpar
        q2theta = - np.sin(d['theta'])*qrock + np.cos(d['theta'])*qpar

    else:
        qrock_prime, qroll, q2theta = saved_q

    if timeit:
        print('Time for computing q: {:.2f} s'.format(time.time()-tic))

    # Convert point cloud into local density function, Resq_i, normalized to 1
    # If the range is set too narrow such that some points fall outside ranges,
    #             the fraction of points outside is returned as ratio_outside
    if saved_q is None:
        index1 = np.floor((qrock_prime + q1_range / 2) / q1_range * (npoints1 - 1)).astype(int)
        index2 = np.floor((qroll + q2_range / 2) / q2_range * (npoints2 - 1)).astype(int)
        index3 = np.floor((q2theta + q3_range / 2) / q3_range * (npoints3 - 1)).astype(int)
    else:
        a1 = 1 / q1_range * (npoints1 - 1)
        b1 = (q1_range / 2 + np.cos(d['theta'])*d['phi']) * a1
        a2 = 1 / q2_range * (npoints2 - 1)
        b2 = (q2_range / 2 + d['chi']) * a2
        a3 = 1 / q3_range * (npoints3 - 1)
        b3 = (q3_range / 2 - np.sin(d['theta'])*d['phi']) * a3
        index1 = np.floor(qrock_prime * a1 + b1).astype(int)
        index2 = np.floor(qroll * a2 + b2).astype(int)
        index3 = np.floor(q2theta * a3 + b3).astype(int)

    # count the total number of outside rays
    outside_ind = ((index1 < 0) | (index1 >= npoints1) |
                   (index2 < 0) | (index2 >= npoints2) |
                   (index3 < 0) | (index3 >= npoints3))
    outside = np.count_nonzero(outside_ind)

    # count the histogram of the 3d voxelized space to estimate Resq_i
    ind = np.stack([index1, index2, index3], axis=-1) # (Nrays, 3)
    ind_inside = ind[np.logical_not(outside_ind), :]  # remove the outside voxels
    # print(ind_inside)

    if timeit:
        print('Time for Monte Carlo ray tracing: {:.2f} s'.format(time.time() - tic))

    # count the elements in each voxel, pad to the shape of [np1 np2 np3]
    Res_qi = np.bincount(np.ravel_multi_index(ind_inside.T, (npoints1, npoints2, npoints3)), minlength=npoints1*npoints2*npoints3).reshape((npoints1, npoints2, npoints3)).astype(float)
    Res_qi = Res_qi / np.max(Res_qi) # normalize to 1
    
    ratio_outside = outside / Nrays

    if timeit:
        print('Time for voxelizing: {:.2f} s'.format(time.time() - tic))

    if plot:
        plot_half_range = 0.0045
        _, ax = plot_res(qrock, qroll, qpar, plot_half_range=plot_half_range, show=False)
        ax.set_xlabel(r'$\hat{q}_{rock}$')
        ax.set_ylabel(r'$\hat{q}_{roll}$')
        ax.set_zlabel(r'$\hat{q}_{par}$')
        ax.set_title('Crystal system')
        plt.show()
        _, ax = plot_res(qrock_prime, qroll, q2theta, plot_half_range=plot_half_range, show=False)
        ax.set_xlabel(r'$\hat{q}_{rock}^\prime$')
        ax.set_ylabel(r'$\hat{q}_{roll}$')
        ax.set_zlabel(r'$\hat{q}_{2\theta}$')
        ax.set_title('Imaging system')
        plt.show()

    if saved_q is None:
        return Res_qi, (qrock_prime, qroll, q2theta)
    else:
        return Res_qi, ratio_outside

def forward(d, Fg_fun, Npixels=50, psize=75e-9, Res_qi=None, plot_im=False, plot_qi=False, timeit=False):
    '''Generate voxelized intensities and the image for DFXM

    Parameters
    ----------
    d : dict
        dictionary for dislocation and instrumental settings
    Fg_fun : function handle
        function for calculating the displacement gradient tensor
    Npixels : int, default 50
        number of pixels in each of y and z directions, set the FOV
    psize : float, default 75e-9
        pixel size in units of m, in the object plane
    Res_qi : array, default None
        pre-computed Res_qi, if None, the Res_qi is calculated
    plot_im : bool
        whether the image are visualized
    plot_qi : bool
        whether the voxelized intensities are visualized
    timeit : bool
        whether print the timing info of the algorithm

    Returns
    -------
    im : array of (Npixels, Npixels)
        image of DFXM given the strain tensor
    qi_field : array of (Npixels, Npixels, Npixels, 3)
        3D voxelized field of intensities
    '''
    Nsub = 2            # multiply 2 to avoid sampling the 0 point, make the grids symmetric over 0
    NN = Nsub*Npixels   # NN^3 is the total number of "rays" (voxels?) probed in the sample
    
    # INPUT instrumental settings, related to direct space resolution function
    if 'psize' in d:
        psize = d['psize'] # pixel size in units of m, in the object plane
    if 'zl_rms' in d:
        zl_rms = d['zl_rms']
    else:
        zl_rms = 0.6E-6/2.35 # rms value of Gaussian beam profile, in m, centered at 0
    if 'two_theta' in d:
        theta_0 = np.deg2rad(d['two_theta']/2) # in rad
    else:
        theta_0 = np.deg2rad(20.73/2) # in rad
    if 'hkl' in d:
        v_hkl = d['hkl']
    else:
        v_hkl = [0, 0, 1]
    if 'TwoDeltaTheta' in d:
        TwoDeltaTheta = d['TwoDeltaTheta']
    else:
        TwoDeltaTheta = 0
    if 'Usg' in d:
        U = d['Usg']
    else:
        U = np.eye(3)
    if 'phi' in d:
        phi = d['phi']
    else:
        phi = 0
    if 'chi' in d:
        chi = d['chi']
    else:
        chi = 0

    # if 'q_ranges' in d:
    #     q1_range, q2_range, q3_range = d['q_ranges']
    # else:
    #     q1_range = q2_range = q3_range = 8e-3
    # if 'npoints' in d:
    #     npoints1, npoints2, npoints3 = d['npoints']
    # else:
    #     npoints1 = npoints2 = npoints3 = 40
    q1_range = q2_range = q3_range = 8e-3
    if 'q1_range' in d:
        q1_range = d['q1_range']
    if 'q2_range' in d:
        q2_range = d['q2_range']
    if 'q3_range' in d:
        q3_range = d['q3_range']
    npoints1 = npoints2 = npoints3 = 40
    if 'npoints1' in d:
        npoints1 = d['npoints1']
    if 'npoints2' in d:
        npoints2 = d['npoints2']
    if 'npoints3' in d:
        npoints3 = d['npoints3']
    if 'Nrays' in d:
        Nrays = d['Nrays']
    else:
        Nrays = 10000000

    if 'x_center' in d:
        x_center = d['x_center']
    else:
        x_center = 0
    if 'y_center' in d:
        y_center = d['y_center']
    else:
        y_center = 0
    if 'z_center' in d:
        z_center = d['z_center']
    else:
        z_center = 0

    if timeit: 
        tic = time.time()

    # Calculate reciprocal-space resolusion function
    if Res_qi is None:
        Res_qi, ratio_outside = res_fn(d, q1_range=q1_range, q2_range=q2_range, q3_range=q3_range, npoints1=npoints1, npoints2=npoints2, npoints3=npoints3, Nrays=Nrays, plot=False)

    # Define the grid of points in the lab system (xl, yl, zl)
    theta = theta_0 + TwoDeltaTheta
    yl_start = -psize*Npixels/2 + psize/(2*Nsub) + y_center # start in yl direction, in units of m, centered at 0
    yl_step = psize/Nsub
    xl_start = ( -psize*Npixels/2 + psize/(2*Nsub) )/np.tan(2*theta) + x_center # start in xl direction, in m, for zl=0
    xl_step = psize/Nsub/np.tan(2*theta)
    zl_start = -0.5*zl_rms*6 + z_center  # start in zl direction, in m, for zl=0
    zl_step = zl_rms*6/(NN-1)

    qi1_start, qi1_step = -q1_range/2, q1_range/(npoints1-1)
    qi2_start, qi2_step = -q2_range/2, q2_range/(npoints2-1)
    qi3_start, qi3_step = -q3_range/2, q3_range/(npoints3-1)

    Q_norm = np.linalg.norm(v_hkl)  # We have assumed B_0 = I (?)
    q_hkl = v_hkl/Q_norm

    # Define the rotation matrices
    mu = theta_0
    M = [[np.cos(mu), 0, np.sin(mu)],
         [0, 1, 0],
         [-np.sin(mu), 0, np.cos(mu)],
    ]
    Omega = np.eye(3)
    Chi = np.eye(3)
    Phi = np.eye(3)
    Gamma = M@Omega@Chi@Phi
    Theta = [[np.cos(theta), 0, np.sin(theta)],
             [0, 1, 0],
             [-np.sin(theta), 0, np.cos(theta)],
    ]

    im = np.zeros([Npixels,Npixels])  # The forward model image
    qi_field = np.zeros([NN,NN,NN,3]) # wave vector function
    
    # The tensor fields Fg, Hg and the vector fields qs, qc, qi are all defined as 3D fields in the lab system

    yl = yl_start + np.arange(NN)*yl_step
    zl = zl_start + np.arange(NN)*zl_step
    xl0= xl_start + np.arange(NN)*xl_step
    rulers = np.array([xl0, yl, zl]) # rulers in the lab system (for plotting)
    # create the 3D grid of points in the lab system, the first dimension is zl, the second is yl, the third is xl
    # YL[:,i,j] == yl; ZL[i,:,j] == zl; XL0[i,j,:] == xl0;
    ZL, YL, XL0 = np.meshgrid(zl, yl, xl0)

    if timeit:
        print('Initialization time: %.2fs'%(time.time() - tic))
    
    XL = XL0 + ZL/np.tan(2*theta)
    PZ = np.exp(-0.5*(ZL/zl_rms)**2)    # Gaussian beam in zl (a thin slice of sample)
    RL = np.stack([XL, YL, ZL], axis=-1)   # (NN,NN,NN,3)
    # Determine the location of the pixel on the detector
    DET_IND_Y = np.round((YL-yl_start)/yl_step).astype(int)//Nsub # np.floor((YL - yl_start)/yl_step/Nsub).astype(int) # THIS ALIGNS WITH yl
    DET_IND_Z = np.round((XL0-xl_start)/xl_step).astype(int)//Nsub # np.floor((XL0- xl_start)/xl_step/Nsub).astype(int) # THIS IS THE OTHER DETECTOR DIRECTION AND FOLLOWS xl BUT WITH MAGNIFICATION (?)
    RS = np.einsum('ji,...j->...i', Gamma, RL) # NB: Gamma inverse Eq. 5
    RG = np.einsum('ji,...j->...i', U, RS)     # NB U inverse, Eq. 7
    Fg = Fg_fun(d, RG[..., 0], RG[..., 1], RG[..., 2]) # calculate the displacement gradient

    # determine the qi for given voxel
    Hg = np.swapaxes(np.linalg.inv(Fg), -1, -2)-np.eye(3)         # Eq. 31
    QS = np.einsum('ij,...jk,k->...i', U, Hg, q_hkl)    # Eq. 32
    QC = QS + np.array([phi - TwoDeltaTheta/2, chi, (TwoDeltaTheta/2)/np.tan(theta_0)])                                         # Eq. 40 (also Eq. 20)
    QI = np.einsum('ij,...j->...i', Theta, QC)          # Eq. 41
    qi_field = np.swapaxes(np.swapaxes(QI, 2, 1), 1, 0) # for plotting, sorted in order x_l,y_l,z_l,:

    # Interpolation in rec. space resolution function.
    IND1 = np.floor( (QI[...,0] - qi1_start)/qi1_step).astype(int)
    IND2 = np.floor( (QI[...,1] - qi2_start)/qi2_step).astype(int)
    IND3 = np.floor( (QI[...,2] - qi3_start)/qi3_step).astype(int)

    if timeit:
        print('Calculate the wave vectors: %.2fs'%(time.time() - tic))

    # Determine intensity contribution from voxel based on rec.space res.function
    PROB = np.zeros((NN,NN,NN))
    IND_IN = ((IND1 >= 0) & (IND1 < npoints1) &
              (IND2 >= 0) & (IND2 < npoints2) &
              (IND3 >= 0) & (IND3 < npoints3)
    )
    # for i,j,k in zip(np.nonzero(IND_IN)[0], np.nonzero(IND_IN)[1], np.nonzero(IND_IN)[2]):
    #     PROB[i,j,k] = Res_qi[IND1[i,j,k],IND2[i,j,k],IND3[i,j,k]]*PZ[i,j,k]
    PROB[IND_IN] = Res_qi[IND1[IND_IN],IND2[IND_IN],IND3[IND_IN]]*PZ[IND_IN]

    # Sum over all pixels in the detector, equivalent to the following loops but faster
    # for i in range(NN):
    #     for j in range(NN):
    #         for k in range(NN):
    #             # im[k//Nsub, i//Nsub] += PROB[i,j,k]
    #             im[DET_IND_Z[i,j,k], DET_IND_Y[i,j,k]] += PROB[i,j,k]
    ravel_ind = np.ravel_multi_index((DET_IND_Z.flatten(), DET_IND_Y.flatten()), (Npixels, Npixels)) # shape (NN**3,), values in range(Npixels**2)
    im = np.bincount(ravel_ind.flatten(), weights=PROB.flatten(), minlength=Npixels*Npixels).reshape(Npixels,Npixels) # shape (Npixels,Npixels)

    if timeit:
        print('Image calculation: %.2fs'%(time.time() - tic))

    if plot_im:
        fig, ax = plt.subplots()
        imax = ax.imshow(im, extent=[xl0.min(), xl0.max(), yl.min(), yl.max()])
        ax.set_title('Image')
        fig.colorbar(imax, ax=ax)
        plt.show()
    if plot_qi:
        pass

    return im, qi_field, rulers

def compute_image_1disl(d, dis_grain_all, char, loc, Npixels, psize, zsize, plot = False, label = None):
    '''
    Generates grid of intensities (image) for a single dislocation
    TO DO: Generalize to more than one dislocation

    Parameters
    ----------
    d : TYPE
        DESCRIPTION.
    char : TYPE
        DESCRIPTION.
    loc : TYPE
        DESCRIPTION.
    Npixels : TYPE
        DESCRIPTION.
    psize : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    theta = d['two_theta']/2*(pi/180)
    theta_p = theta + d['TwoDeltaTheta']
    
    image_lab = get_image_lab(d)
    # define lattice vector
    Q_norm = np.linalg.norm(d['hkl']) # We have assumed B_0 = I
    q_hkl = d['hkl']/Q_norm
    Chi, Phi, Mu = get_rot_matrices(d)
    
    lab_sample = np.matmul(np.matmul(Mu,Omega), np.matmul(Chi,Phi))
    image_sample = np.matmul(np.transpose(image_lab[0,:]), lab_sample)
    dis_lab = get_dis_lab(d, dis_grain_all, char)
    dis_sample = np.matmul(dis_lab,lab_sample)
    
    xlab_vec, ylab_vec, zlab_vec = define_voxel(d, Npixels, psize, zsize)
    
    intensity_fxn = res_fxn_q(d)
    
    
    image = np.zeros([Npixels, Npixels])
    for ii in range(Npixels):
        y_lab = ylab_vec[ii]
        for jj in range(Npixels):
            z_lab = zlab_vec[jj]
            intensity = 0
            for x_lab in xlab_vec:
                #x_lab = x_lab + z_lab/tan(2*theta)
                                
                #Why is x_lab modified twice?
                r_lab = [[x_lab + z_lab/tan(2*theta_p)],[y_lab],[z_lab]]
                
                r_dis = np.matmul(dis_lab[0:2,:],r_lab)
                xd = r_dis[0] - loc[0]
                yd = r_dis[1] - loc[1]
                
                Fdis = edge.get_disl_strain_tensor_single(d, xd, yd)
                
                Fsample = np.matmul(np.matmul(np.transpose(dis_sample), Fdis),\
                                    dis_sample)
                
                    
                qi_s = np.matmul(np.linalg.inv(np.transpose(Fsample)), q_hkl) -\
                    q_hkl

                qi_s = qi_s + [[d['phi'] - d['TwoDeltaTheta']/2], [d['chi']],\
                                [(d['TwoDeltaTheta']/2)/tan(theta_p)]]
                
                intensity = intensity +\
                    intensity_fxn['Qrock_prime']['intensity_fxn'](qi_s[0])[0]
                
            image[ii, jj] = intensity
    
    def generate_image(image, imgfilter = None, sigma=0.2):     
        # if imgfilter == 'gauss':
        #     image = gaussian_filter(image, sigma=sigma)
            
        fig, ax = plt.subplots(figsize = (5,5))
        yim_vec = ylab_vec/1E3
        zim_vec = zlab_vec/1E3/tan(2 * d['theta'])
        ax.set_aspect(1)
        #ax.set_aspect(zim_vec[-1]/yim_vec[-1], 'box')
        # val = ax.imshow(image, aspect=zim_vec[-1]/yim_vec[-1], extent =\
        #           (yim_vec[0], yim_vec[-1], zim_vec[0], zim_vec[-1]))
        val = ax.imshow(image, aspect = 0.8, extent =\
                   [yim_vec[0], yim_vec[-1], zim_vec[0], zim_vec[-1]])
        
        fig.colorbar(val)
        plt.xlabel(r'$z$ ($\mu$m)')
        plt.ylabel(r'$y$ ($\mu$m)')
        plt.savefig('figures/dis' + str(char) + label + '.pdf', bbox_inches='tight')
            
    if plot:
        generate_image(image)
                    
    return image
                 
    
        
        
if __name__ == '__main__':
    #char = 5
    loc = [[0], [0], [0]]
    Npixels = 50
    psize = 75 #in nm
    zsize = 600# in nm
    dis_grain_all = edge.return_dis_grain_matrices()
    Chi, Phi, Mu = get_rot_matrices(res_dict)
    q_fxn = res_fxn_q(res_dict)
    xlab_vec, ylab_vec, zlab_vec = define_voxel(res_dict, Npixels, psize, zsize)
    
    for char in range(12):
        image = compute_image_1disl(res_dict, dis_grain_all, char, loc, Npixels, psize, zsize, plot = True, label = '_edge_fcc_2')
    
    