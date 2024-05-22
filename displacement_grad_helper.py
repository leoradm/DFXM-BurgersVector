# -----------------------------------------------------------------------
# Functions for calculating the displacement gradient tensor of a given dislocation configuration
# Date: 2023-05-26
# Original MATLAB code: Nicolas Bertin (bertin1@llnl.gov)
# Python translation: Yifan Wang (yfwang09@stanford.edu)
# -----------------------------------------------------------------------

import numpy as np

def displacement_gradient_optimized(NU, b, rn, links, r, a):
    ''' Calculate the displacement gradient tensor of a dislocation configuration'''
    pass

def displacement_gradient_seg_matlab(NU, b, r1, r2, r, a):
    ''' Calculate the displacement gradient tensor of a dislocation segment
    Direct translation from the MATLAB code displacement_gradient_seg.m

    Parameters
    ----------
    NU : float
        Poisson's ratio
    b : numpy array
        Burgers vector
    r1 : numpy array
        First endpoint of the dislocation segment
    r2 : numpy array
        Second endpoint of the dislocation segment
    r : numpy array
        Observation point
    a : float
        Non-singular radius
    Returns
    -------
    dudx : numpy array
        Displacement gradient tensor
    ''' 
    dudx = np.zeros((3, 3))

    t = r2 - r1
    t = t/np.linalg.norm(t)
    R = r1 - r
    dr = np.dot(R, t)
    x0 = r1 - dr*t
    d = R - dr*t
    s1 = np.dot(r1 - x0, t)
    s2 = np.dot(r2 - x0, t)

    a2 = a**2
    d2 = np.dot(d, d)
    da2 = d2 + a2
    da2inv = 1/da2
    Ra1 = np.sqrt(s1**2 + da2)
    Ra2 = np.sqrt(s2**2 + da2)
    Ra1inv = 1/Ra1
    Ra1inv3 = Ra1inv**3
    Ra2inv = 1/Ra2
    Ra2inv3 = Ra2inv**3
    # print('r', r, 'Ra1', Ra1, 'Ra2', Ra2)

    J03 = da2inv*(s2*Ra2inv - s1*Ra1inv)
    J13 = -Ra2inv + Ra1inv
    J15 = -1/3*(Ra2inv3 - Ra1inv3)
    J25 = 1/3*da2inv*(s2**3*Ra2inv3 - s1**3*Ra1inv3)
    J05 = da2inv*(2*J25 + s2*Ra2inv3 - s1*Ra1inv3)
    J35 = 2*da2*J15 - s2**2*Ra2inv3 + s1**2*Ra1inv3
    # print('r', r, 'J03', J03, 'J13', J13, 'J15', J15, 'J25', J25, 'J05', J05, 'J35', J35)

    delta = np.eye(3)
    A = 3*a2*d*J05 + 2*d*J03 + 3*a2*t*J15 + 2*t*J13
    # print('r', r, 'A', A) 

    B1 = (np.einsum('mj,l->jlm', delta, d) + np.einsum('jl,m->jlm', delta, d) + np.einsum('lm,j->jlm', delta, d)) * J03
    B2 = (np.einsum('mj,l->jlm', delta, t) + np.einsum('jl,m->jlm', delta, t) + np.einsum('lm,j->jlm', delta, t)) * J13
    B3 = -3*np.einsum('m,j,l->jlm', d, d, d) * J05
    B4 = -3*(np.einsum('m,j,l->jlm', d, d, t) + np.einsum('m,j,l->jlm', d, t, d) + np.einsum('m,j,l->jlm', t, d, d)) * J15
    B5 = -3*(np.einsum('m,j,l->jlm', d, t, t) + np.einsum('m,j,l->jlm', t, d, t) + np.einsum('m,j,l->jlm', t, t, d)) * J25
    B6 = -3*np.einsum('m,j,l->jlm', t, t, t) * J35
    B  = B1 + B2 + B3 + B4 + B5 + B6

    Ab = np.einsum('i,j,k->ijk', A, t, b)
    
    U1 = np.zeros((3, 3))
    U2 = np.zeros((3, 3))
    U3 = np.zeros((3, 3))
    for l in range(3):
        U1[l, 0] = Ab[2, 1, l] - Ab[1, 2, l]
        U1[l, 1] = Ab[0, 2, l] - Ab[2, 0, l]
        U1[l, 2] = Ab[1, 0, l] - Ab[0, 1, l]
        U2[0, l] = Ab[l, 2, 1] - Ab[l, 1, 2]
        U2[1, l] = Ab[l, 0, 2] - Ab[l, 2, 0]
        U2[2, l] = Ab[l, 1, 0] - Ab[l, 0, 1]
        for m in range(3):
            U3[m, l] = (B[1, l, m]*t[2] - B[2, l, m]*t[1])*b[0] \
                     + (B[2, l, m]*t[0] - B[0, l, m]*t[2])*b[1] \
                     + (B[0, l, m]*t[1] - B[1, l, m]*t[0])*b[2]

    dudx = -1/8/np.pi*(U1 + U2 + 1/(1 - NU) * U3)
    return dudx

def displacement_gradient_structure_matlab(rn, links, NU, a, r, test=False):
    '''Computes the non-singular displacement gradient produced by the dislocation structure.

    Parameters
    ----------
    rn : numpy array
        Position of the nodes
    links : numpy array
        Connectivity of the nodes
    NU : float
        Poisson's ratio
    a : float
        Non-singular radius
    r : numpy array
        Observation point
    Returns
    -------
    dudx : numpy array
        Displacement gradient tensor
    '''
    
    dudx = np.zeros((r.shape[0], 3, 3))
    if not test:
        # it = np.nditer(r, flags=['multi_index'])
        for j in range(r.shape[0]):
            for i in range(links.shape[0]):
                n1 = links[i, 0].astype(int)
                n2 = links[i, 1].astype(int)
                r1 = rn[n1, :]
                r2 = rn[n2, :]
                b = links[i, 2:5]
                dudx[j, :] += displacement_gradient_seg_matlab(NU, b, r1, r2, r[j, :], a)
    
    return dudx

def displacement_seg_ns(x1, x2, b, NU, a, P):
    '''Computes the non-singular displacement produced by a dislocation segment.'''
    a2 = a**2
    t = x2 - x1
    t = t/np.linalg.norm(t)
    x0 = x1 + np.dot(P - x1, t)*t
    d = P - x0
    d2 = np.dot(d, d)
    s1 = np.dot(x1 - x0, t)
    s2 = np.dot(x2 - x0, t)

    Ra1 = np.sqrt(d2 + s1**2 + a2)
    Ra2 = np.sqrt(d2 + s2**2 + a2)

    J01 = np.log(Ra2 + s2) - np.log(Ra1 + s1)
    J03 = s2/(d2 + a2)/Ra2 - s1/(d2 + a2)/Ra1
    J13 = -1/Ra2 + 1/Ra1
    J23 = J01 - s2/Ra2 + s1/Ra1

    A = np.cross(t, b)
    A = (a2*J03 + 2*J01)*A

    delta = np.eye(3)
    C = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            C[i, j] = delta[i, j]*J01 - d[i]*d[j]*J03 + (d[i]*t[j] + t[i]*d[j])*J13 - t[i]*t[j]*J23

    B = np.zeros(3)
    for i in range(3):
        B[i] = t[0]*C[1, i]*b[2] + t[1]*C[2, i]*b[0] + t[2]*C[0, i]*b[1] \
             - t[2]*C[1, i]*b[0] - t[0]*C[2, i]*b[1] - t[1]*C[0, i]*b[2]
    
    u = 1/8/np.pi*(A + B/(1 - NU))
    return u

def triangular_displacement_ns(A, B, C, b, n, NU, a, P):
    '''Computes the non-singular displacement field produced by a triangular dislocation loop.

    Parameters
    ----------
    A : numpy array
        First vertex of the triangle
    B : numpy array
        Second vertex of the triangle
    C : numpy array
        Third vertex of the triangle
    b : numpy array
        Burgers vector
    n : numpy array
        Normal vector
    NU : float
        Poisson's ratio
    a : float
        Non-singular radius
    P : numpy array
        Observation point
    Returns
    -------
    u : numpy array
        Displacement vector
    '''
    
    Tab = B - A; Tab = Tab/np.linalg.norm(Tab)
    Tbc = C - B; Tbc = Tbc/np.linalg.norm(Tbc)
    Tca = A - C; Tca = Tca/np.linalg.norm(Tca)

    # n0 is the normal counter clockwise
    n0 = np.cross(Tab, -Tca)
    n0 = n0/np.linalg.norm(n0)

    if np.abs(np.dot(n0,n)-1) > 1e-5:
        raise ValueError('Triangular loop does not lie in its plane.')
    
    Pa = A - P
    Pb = B - P
    Pc = C - P
    Ra = np.linalg.norm(Pa)
    Rb = np.linalg.norm(Pb)
    Rc = np.linalg.norm(Pc)
    La = Pa/Ra
    Lb = Pb/Rb
    Lc = Pc/Rc
    # print(A, P, Pa, Ra, La)

    a0 = np.arccos(np.dot(Lb, Lc))
    b0 = np.arccos(np.dot(La, Lc))
    c0 = np.arccos(np.dot(La, Lb))
    s0 = 0.5*(a0 + b0 + c0)

    tans = np.tan(s0/2)
    tana = np.tan((s0 - a0)/2)
    tanb = np.tan((s0 - b0)/2)
    tanc = np.tan((s0 - c0)/2)
    tanp = tans*tana*tanb*tanc
    tanp = np.max(tanp, 0)

    if np.abs(np.dot(La, n0)) < 1e-10:
        sgn = 0
    else:
        sgn = np.sign(np.dot(La, n0))
    Omega = -sgn*(4*np.arctan(np.sqrt(tanp)))

    # Solid angle
    u = -Omega/4/np.pi*b

    # Line integrals
    u += displacement_seg_ns(A, B, b, NU, a, P)
    u += displacement_seg_ns(B, C, b, NU, a, P)
    u += displacement_seg_ns(C, A, b, NU, a, P)

    return u

def displacement_structure(rn, links, NU, a, r):
    '''Computes the non-singular displacement field produced by the dislocation structure.
    This function only works for triangular loops.
    (Based on Fivel and Depres, Phil. Mag., 2014).

    Parameters
    ----------
    rn : numpy array
        Position of the nodes
    links : numpy array
        Connectivity of the nodes
    NU : float
        Poisson's ratio
    a : float
        Non-singular radius
    r : numpy array
        Observation point
    Returns
    -------
    u : numpy array
        Displacement vector
    '''
    
    if rn.shape[0] > 3:
        raise ValueError('This function only works for triangular loops.')
    
    u = np.zeros((r.shape[0], 3))
    # it = np.nditer(r, flags=['multi_index'])

    # Loop center and loop plane normal
    rc = np.mean(rn, axis=0)
    n = np.cross(rn[1, :] - rn[0, :], rn[2, :] - rn[0, :])
    n = n/np.linalg.norm(n)

    for j in range(r.shape[0]):
        rj = r[j, :]
        for i in range(links.shape[0]):
            n1 = links[i, 0].astype(int)
            n2 = links[i, 1].astype(int)
            r1 = rn[n1, :]
            r2 = rn[n2, :]
            b = links[i, 2:5]
            
            # Check that each segment is comprised in the plane
            if np.dot(r2 - r1, n) > 1e-5:
                raise ValueError('The loop is not planar.')
            
            # Compute displacement
            # print('j, r[j]', j, rj)
            u[j, :] += triangular_displacement_ns(r1, r2, rc, b, n, NU, a, rj)

    return u
