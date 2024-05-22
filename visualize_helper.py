#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:00:19 2023

@author: yfwang09


Visualization helper functions for debugging and testing.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_2d_slice_z(F, extent=[-1, 1, -1, 1, -1, 1], nslice=9, show=True):
    '''Plot 2D slices in z direction. (unfinished)

    Parameters
    ----------
    F : ndarray
        3D array to be plotted.
    extent : list, optional
        Extent of the plot. The default is [-1, 1, -1, 1, -1, 1].
    nslice : int, optional
        Number of slices to plot. The default is 9.
    show : bool, optional
        Whether to show the plot. The default is True.

    Returns
    -------
    figax : tuple
        Figure and axes of the plot.
    '''
    npoints3 = F.shape[2]
    iz_slices = np.linspace(0, npoints3-1, nslice+2).astype(int)
    fig, axs = plt.subplots(3, 3, figsize=(10,6), sharex=True, sharey=True)
    for i in range(1, iz_slices.size-1):
        ind, iz = i-1, iz_slices[i]
        # zval = (iz/(npoints3-1)*2-1)*extent[5]/2
        ax = axs[ind//3, ind%3]
        imax = ax.imshow(F[:,:,iz], extent=extent[:4], vmin=0, vmax=1)
        # ax.set_xticks(ax.get_yticks())
        # ax.set_title(r'$\hat{q}_{par}$=%.4f'%zval, y=0.8, va='top', color='w')
        # if ind//3 == 2:
        #     ax.set_xlabel(r'$\hat{q}_{roll}$')
        # if ind%3 == 0:
        #     ax.set_ylabel(r'$\hat{q}_{rock}$')
        # ax.ticklabel_format(style='sci', scilimits=(0,0))
    pass

def visualize_res_fn_slice_z(d, Res_qi, plot_2d=True, plot_3d=True, show=True):
    '''Visualize the resolution function in 2D and 3D slices in z direction.
    
    Parameters
    ----------
    d : dict
        Dictionary of parameters.
    Res_qi : ndarray
        Resolution function.
    plot_2d : bool, optional
        Whether to plot 2D slices. The default is True.
    plot_3d : bool, optional
        Whether to plot 3D slices. The default is True.
    show : bool, optional
        Whether to show the plots. The default is True.

    Returns
    -------
    figax2d : tuple
        Figure and axes of 2D plots, if plot_2d is True, None otherwise.
    figax3d : tuple
        Figure and axes of 3D plots, if plot_3d is True, None otherwise.
    '''

    lb, ub = -d['q3_range']/2, d['q3_range']/2
    npoints3 = d['npoints3']
    figax2d = None
    figax3d = None
    if plot_2d:
        nslice = 9
        iz_slices = np.linspace(0, npoints3-1, nslice+2).astype(int)

        fig, axs = plt.subplots(3, 3, figsize=(10,6), sharex=True, sharey=True)
        q_extent = [-d['q2_range']/2, d['q2_range']/2, -d['q1_range']/2, d['q1_range']/2]

        for i in range(1, iz_slices.size-1):
            ind, iz = i-1, iz_slices[i]
            zval = (iz/(npoints3-1)*2-1)*d['q3_range']/2
            # print(iz, zval)
            ax = axs[ind//3, ind%3]
            imax = ax.imshow(Res_qi[:,:,iz], extent=q_extent, vmin=0, vmax=1)
            ax.set_xticks(ax.get_yticks())
            ax.set_title(r'$\hat{q}_{par}$=%.4f'%zval, y=0.8, va='top', color='w')
            if ind//3 == 2:
                ax.set_xlabel(r'$\hat{q}_{roll}$')
            if ind%3 == 0:
                ax.set_ylabel(r'$\hat{q}_{rock}$')
            ax.ticklabel_format(style='sci', scilimits=(0,0))
        ax.set_xlim(q_extent[:2])
        ax.set_ylim(q_extent[2:])
        cbar_ax = fig.colorbar(imax, ax=axs)
        cbar_ax.set_label(r'Intensity (a.u.)')
        fig.suptitle(r'Resolution function: $\phi=%.4f,\chi=%.4f,\omega=%.4f$'%(
            d['phi'], d['chi'], d['omega']
        ))
        figax2d = (fig, axs)
        if show:
            plt.show()

    if plot_3d:
        x = d['q1_range']*(np.arange(d['npoints1'])/(d['npoints1']-1)*2-1)/2
        y = d['q2_range']*(np.arange(d['npoints2'])/(d['npoints2']-1)*2-1)/2
        yy, xx = np.meshgrid(y, x)

        nslice = 5
        iz_slices = np.linspace(0, npoints3-1, nslice+2).astype(int)

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([lb, lb, ub, ub], [ub, ub, ub, ub], [lb, ub, ub, lb], 'k')
        ax.plot([lb, lb, ub, ub], [lb, ub, ub, lb], [lb, lb, lb, lb], 'k')

        for i in range(1, iz_slices.size-1):
            ind, iz = i-1, iz_slices[i]
            zloc = (iz/(npoints3-1)*2-1)*d['q3_range']/2    # qpar location of the plane
            Fg_z = Res_qi[:,:,iz]                   # values of the plane
            surf = ax.plot_surface(xx, yy, np.ones_like(xx)*zloc, linewidth=0, edgecolor="None", facecolors=plt.cm.viridis(Fg_z), alpha=0.5)
            
        ax.plot([lb, lb, ub, ub], [lb, lb, lb, lb], [ub, lb, lb, ub], 'k')
        ax.plot([lb, lb, ub, ub], [ub, lb, lb, ub], [ub, ub, ub, ub], 'k')
        ax.set_ylabel(r'$\hat{q}_{roll}(\times10^{-3})$')#, fontsize=fs)
        ax.set_xlabel(r'$\hat{q}_{rock}(\times10^{-3})$')#, fontsize=fs)
        ax.set_zlabel(r'$\hat{q}_{par}(\times10^{-3})$')#, fontsize=fs)
        ax.ticklabel_format(style='sci', scilimits=(0,0))
        ax.set_title(r'Reciprocal space resolution function Res(q)')#, fontsize=fs)
        cax = fig.colorbar(surf, shrink=0.6)
        figax3d = (fig, ax)
        if show:
            plt.show()
    return figax2d, figax3d

def visualize_im_qi(forward_dict, im, qi, rulers, vlim_im=[None, None], vlim_qi=[None, None], deg=False, show=True):
    ''' Visualize the simulated image and the reciprocal space wave vectors

    Parameters
    ----------
    forward_dict : dict
        The dictionary containing the forward model parameters.
    im : 2D array
        The simulated image.
    qi : 3D array
        The reciprocal space wave vectors.
    rulers : 3xN array
        The rulers for the reciprocal space wave vectors.
    vlim_im : list, optional
        The intensity limits for the image. The default is [None, None].
    vlim_qi : list, optional
        The intensity limits for the reciprocal space wave vectors. The default is [None, None].
    deg : bool, optional
        Whether the angles are in degrees. The default is False.
    show : bool, optional
        Whether to show the figure. The default is True.

    Returns
    -------
    fig_im : matplotlib figure
        The figure for the image.
    ax_im : matplotlib axis
        The axis for the image.
    fig_qi_z : matplotlib figure
        The figure for the reciprocal space wave vectors in the z direction.
    axs_qi_z : matplotlib axis
        The axis for the reciprocal space wave vectors in the z direction.
    fig_qi_y : matplotlib figure
        The figure for the reciprocal space wave vectors in the y direction.
    axs_qi_y : matplotlib axis
        The axis for the reciprocal space wave vectors in the y direction.
    '''
    xl, yl, zl = rulers[0,:], rulers[1,:], rulers[2,:]
    if im is not None:
        vmin, vmax = vlim_im
        # Visualize the simulated image
        fig, ax = plt.subplots()
        imax = ax.imshow(im.T, extent=[xl.min(), xl.max(), yl.min(), yl.max()], vmin=vmin, vmax=vmax)
        if deg:
            ax.set_title(r'Simulated DFXM Image: $\phi = %.4f^\circ$, $\chi = %.4f^\circ$, $\Delta 2\theta = %.4f^\circ$'%tuple(np.rad2deg([forward_dict['phi'], forward_dict['chi'], forward_dict['TwoDeltaTheta']])), loc='right')
        else:
            ax.set_title(r'Simulated DFXM Image: $\phi = %.4f$, $\chi = %.4f$, $\Delta 2\theta = %.4f$'%(forward_dict['phi'], forward_dict['chi'], forward_dict['TwoDeltaTheta']), loc='right')
        ax.set_xlabel('x (m)', fontsize=14)
        ax.set_ylabel('y (m)', fontsize=14)
        cax = fig.colorbar(imax, ax=ax, orientation='horizontal')
        cax.set_label('Intensity (a.u.)', fontsize=14)
        cax.ax.tick_params(labelsize=12)
        fig_im, ax_im = fig, ax
    else:
        fig_im, ax_im = None, None

    if qi is not None:
        vmin, vmax = vlim_qi
        # Visualize the reciprocal space wave vector ql in the (x,y,z=0) plane
        ind_z = zl.size//2
        subs = ['x', 'y', 'z']
        fig, axs = plt.subplots(3, 1, figsize=(12,9), sharex=True, sharey=True)
        for i in range(3):
            ax = axs[i]
            imax = ax.imshow(qi[:,:,ind_z,i].T, extent=[xl.min(), xl.max(), yl.min(), yl.max()], vmin=vmin, vmax=vmax)
            ax.set_title(r'$q^{\ell}_{%s}$, (x, y)-plane at z = 0'%(subs[i]), loc='right')
            ax.set_ylabel('$y_{\ell}$ (m)')
        ax.set_xlabel('$x_{\ell}$ (m)')
        # create a shared colorbar on the right side
        fig.subplots_adjust(right=0.99)
        y0, y1 = axs[2].get_position().y0, axs[0].get_position().y1
        cbar_ax = fig.add_axes([0.8, y0, 0.05, y1-y0])
        fig.colorbar(imax, cax=cbar_ax)
        # change the colorbar ticks to be in units of 1e-4
        cbar_ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        cbar_ax.set_ylabel(r'$q^{\ell}$ (m$^{-1}$)')
        fig_qi_z, axs_qi_z = fig, axs

        # Visualize the reciprocal space wave vector ql in the (x,y=0,z) plane
        ind_y = yl.size//2
        subs = ['x', 'y', 'z']
        fig, axs = plt.subplots(3, 1, figsize=(12,6), sharex=True, sharey=True)
        for i in range(3):
            ax = axs[i]
            imax = ax.imshow(qi[:,ind_y,:,i].T, extent=[xl.min(), xl.max(), zl.min(), zl.max()], vmin=vmin, vmax=vmax)
            ax.set_title(r'$q^{\ell}_{%s}$, (x, z)-plane at y = 0'%(subs[i]), loc='right')
            ax.set_ylabel('$z_{\ell}$ (m)')
        ax.set_xlabel('$x_{\ell}$ (m)')
        # create a shared colorbar on the right side
        fig.subplots_adjust(right=0.75)
        y0, y1 = axs[2].get_position().y0, axs[0].get_position().y1
        cbar_ax = fig.add_axes([0.8, y0, 0.05, y1-y0])
        fig.colorbar(imax, cax=cbar_ax)
        # change the colorbar ticks to be in units of 1e-4
        cbar_ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        cbar_ax.set_ylabel(r'$q^{\ell}$ (m$^{-1}$)')
        fig_qi_y, axs_qi_y = fig, axs
    else:
        fig_qi_z, axs_qi_z = None, None
        fig_qi_y, axs_qi_y = None, None

    if show:
        plt.show()

    return (fig_im, ax_im, fig_qi_z, axs_qi_z, fig_qi_y, axs_qi_y)
