"""This module contains functions for doing complex Gaussian math.
Everything is hard coded for adiabatic/diabatic representation"""

import math
import cmath
import numpy as np


def overlap_nuc(positions_i, positions_j, momenta_i, momenta_j, widths_i,
                widths_j, numdims):
    """Compute the overlap of two nuclear TBFs i and j
    (electronic part not included)"""

    overlap = 1.0

    for idim in range(numdims):
        pos_i_1d = positions_i[idim]
        pos_j_1d = positions_j[idim]
        mom_i_1d = momenta_i[idim]
        mom_j_1d = momenta_j[idim]
        width_i_1d = widths_i[idim]
        width_j_1d = widths_j[idim]
        overlap *= overlap_nuc_1d(pos_i_1d, pos_j_1d, mom_i_1d, mom_j_1d,
                                  width_i_1d, width_j_1d)

    return overlap


def overlap_nuc_1d(pos_i, pos_j, mom_i, mom_j, width_i, width_j):
    """Compute 1-dimensional nuclear overlaps"""

    c1i = (complex(0.0, 1.0))
    delta_x = pos_i - pos_j
    p_diff = mom_i - mom_j
    osmwid = 1.0 / (width_i + width_j)

    xrarg = osmwid * (width_i*width_j*delta_x*delta_x + 0.25*p_diff*p_diff)
    if xrarg < 10.0:
        gmwidth = math.sqrt(width_i*width_j)
        ctemp = (mom_i*pos_i - mom_j*pos_j)
        ctemp = ctemp - osmwid * (width_i*pos_i + width_j*pos_j) * p_diff
        cgold = math.sqrt(2.0 * gmwidth * osmwid)
        cgold = cgold * math.exp(-1.0 * xrarg)
        cgold = cgold * cmath.exp(ctemp * c1i)
    else:
        cgold = 0.0

    return cgold


def kinetic_nuc(positions_i, positions_j, momenta_i, momenta_j, widths_i,
                widths_j, masses_i, numdims):
    """compute the kinetic energy matrix element between two nuclear TBFs"""

    overlap_1d = np.zeros(numdims, dtype=np.complex128)
    kin_e_1d = np.zeros(numdims, dtype=np.complex128)

    for idim in range(numdims):
        pos_i_1d = positions_i[idim]
        pos_j_1d = positions_j[idim]
        mom_i_1d = momenta_i[idim]
        mom_j_1d = momenta_j[idim]
        width_i_1d = widths_i[idim]
        width_j_1d = widths_j[idim]
        mass_i_1d = masses_i[idim]

        kin_e_1d[idim] = 0.5 * kinetic_nuc_1d(pos_i_1d, pos_j_1d, mom_i_1d,
                                              mom_j_1d, width_i_1d,
                                              width_j_1d) / mass_i_1d
        overlap_1d[idim] = overlap_nuc_1d(pos_i_1d, pos_j_1d, mom_i_1d,
                                          mom_j_1d, width_i_1d,
                                          width_j_1d)

    kin_e_ij = 0.0
    for idim in range(numdims):
        Ttmp = kin_e_1d[idim]
        for jdim in range(numdims):
            if jdim != idim:
                Ttmp *= overlap_1d[jdim]
        kin_e_ij += Ttmp

    return kin_e_ij


def kinetic_nuc_1d(pos_i, pos_j, mom_i, mom_j, width_i, width_j):
    """compute 1-dimensional nuclear kinetic energy matrix elements"""

    c1i = (complex(0.0, 1.0))
    p_sum = mom_i + mom_j
    delta_x = pos_i - pos_j
    d_ke_r_fac = width_i + 0.25 * p_sum * p_sum\
        - width_i * width_i * delta_x * delta_x
    d_ke_i_fac = width_i * delta_x * p_sum
    olap = overlap_nuc_1d(pos_i, pos_j, mom_i, mom_j, width_i, width_j)
    kinetic = (d_ke_r_fac + c1i * d_ke_i_fac) * olap

    return kinetic


def Sdot_nuc(positions_i, positions_j, momenta_i, momenta_j, widths_i,
             widths_j, forces_j, masses_i, numdims):
    """Compute the Sdot matrix element between two nuclear TBFs"""

    c1i = (complex(0.0, 1.0))

    overlap = overlap_nuc(positions_i, positions_j, momenta_i, momenta_j,
                          widths_i, widths_j, numdims)

    delta_r = positions_i - positions_j
    mom_sum = momenta_i + momenta_j
    mom_diff = momenta_i - momenta_j
    o4wj = 0.25 / widths_j
    Cdbydr = widths_j * delta_r - (0.5 * c1i) * mom_sum
    Cdbydp = o4wj * mom_diff + (0.5 * c1i) * delta_r
    Ctemp1 = Cdbydr * momenta_j / masses_i + Cdbydp * forces_j
    Ctemp = np.sum(Ctemp1)
    Sdot_ij = Ctemp * overlap

    return Sdot_ij
