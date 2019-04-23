# this module contains functions for doing complex Gaussian math.  Right
# now everything is hard coded for adiabatic/diabatic representation, but
# it shouldn't be hard to modify for DGAS

import sys
import types
import math
import cmath
import numpy as np
from pyspawn.fmsobj import fmsobj
from pyspawn.traj import traj


def overlap_nuc(positions_i, positions_j, momenta_i, momenta_j, widths_i,
                widths_j, numdims):
    """Compute the overlap of two nuclear TBFs (electronic part not included)"""

    Sij = 1.0

    for idim in range(numdims):
        xi = positions_i[idim]
        xj = positions_j[idim]
        di = momenta_i[idim]
        dj = momenta_j[idim]
        xwi = widths_i[idim]
        xwj = widths_j[idim]
        Sij *= overlap_nuc_1d(xi, xj, di, dj, xwi, xwj)

    return Sij


def overlap_nuc_1d(xi, xj, di, dj, xwi, xwj):
    """Compute 1-dimensional nuclear overlaps"""

    c1i = (complex(0.0, 1.0))
    deltax = xi - xj
    pdiff = di - dj
    osmwid = 1.0 / (xwi + xwj)

    xrarg = osmwid * (xwi*xwj*deltax*deltax + 0.25*pdiff*pdiff)
    if (xrarg < 10.0):
        gmwidth = math.sqrt(xwi*xwj)
        ctemp = (di*xi - dj*xj)
        ctemp = ctemp - osmwid * (xwi*xi + xwj*xj) * pdiff
        cgold = math.sqrt(2.0 * gmwidth * osmwid)
        cgold = cgold * math.exp(-1.0 * xrarg)
        cgold = cgold * cmath.exp(ctemp * c1i)
    else:
        cgold = 0.0

    return cgold


def kinetic_nuc(positions_i, positions_j, momenta_i, momenta_j, widths_i,
                widths_j, masses_i, numdims):
    """compute the kinetic energy matrix element between two nuclear TBFs"""

    S1D = np.zeros(numdims, dtype=np.complex128)
    T1D = np.zeros(numdims, dtype=np.complex128)

    for idim in range(numdims):
        xi = positions_i[idim]
        xj = positions_j[idim]
        di = momenta_i[idim]
        dj = momenta_j[idim]
        xwi = widths_i[idim]
        xwj = widths_j[idim]
        m = masses_i[idim]

        T1D[idim] = 0.5 * kinetic_nuc_1d(xi, xj, di, dj, xwi, xwj) / m
        S1D[idim] = overlap_nuc_1d(xi, xj, di, dj, xwi, xwj)

    Tij = 0.0
    for idim in range(numdims):
        Ttmp = T1D[idim]
        for jdim in range(numdims):
            if jdim != idim:
                Ttmp *= S1D[jdim]
        Tij += Ttmp

    return Tij


def kinetic_nuc_1d(xi, xj, di, dj, xwi, xwj):
    """compute 1-dimensional nuclear kinetic energy matrix elements"""
    
    c1i = (complex(0.0, 1.0))
    psum = di + dj
    deltax = xi - xj
    dkerfac = xwi + 0.25 * psum * psum - xwi * xwi * deltax * deltax
    dkeifac = xwi * deltax * psum
    olap = overlap_nuc_1d(xi, xj, di, dj, xwi, xwj)
    kinetic = (dkerfac + c1i * dkeifac) * olap

    return kinetic


def Sdot_nuc(positions_i, positions_j, momenta_i, momenta_j, widths_i,
             widths_j, forces_j, masses_i, numdims):
    """Compute the Sdot matrix element between two nuclear TBFs"""
    
    c1i = (complex(0.0, 1.0))

    Sij = overlap_nuc(positions_i, positions_j, momenta_i, momenta_j, widths_i,
                widths_j, numdims)

    delta_r = positions_i - positions_j
    psum = momenta_i + momenta_j
    pdiff = momenta_i - momenta_j
    o4wj = 0.25 / widths_j
    Cdbydr = widths_j * delta_r - (0.5 * c1i) * psum
    Cdbydp = o4wj * pdiff + (0.5 * c1i) * delta_r
    Ctemp1 = Cdbydr * momenta_j / masses_i + Cdbydp * forces_j
    Ctemp = np.sum(Ctemp1)    
    Sdot_ij = Ctemp * Sij
    
    return Sdot_ij
