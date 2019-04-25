import numpy as np


def construct_ham_lin_slope(x, numstates):
    """1d potential 1 state intersecting other numstates at an angle"""

    k = 0.005  # off-diagonal coupling matrix elements
    slope_1 = 0.25  # slope 1
    slope_2 = 0.025  # slope 2
    delta = 0.01  # gap between diabatic states

    H_elec = np.zeros((numstates, numstates))
    Hx = np.zeros((numstates, numstates))
    H_elec[0, 0] = slope_1 * (-x)
    Hx[0, 0] = -slope_1

    for n in range(numstates - 1):
        if n < 8:  # this if statement adds a gap between middle states
            H_elec[n + 1, n + 1] = slope_2*x - n*delta
            if n != 0:
                H_elec[0, n + 1] = k
                H_elec[n + 1, 0] = k
            else:
                H_elec[0, n + 1] = k  # / 5
                H_elec[n + 1, 0] = k  # / 5
            Hx[n + 1, n + 1] = slope_2

        else:
            H_elec[n + 1, n + 1] = slope_2*x - n*delta  # - 0.08
            if n != 6 and n != 7 and n != 5:
                H_elec[0, n+1] = k
                H_elec[n+1, 0] = k
            else:
                H_elec[0, n + 1] = k  # / 5
                H_elec[n + 1, 0] = k  # / 5
            Hx[n + 1, n + 1] = slope_2

    Force = [Hx]

    return H_elec, Force
