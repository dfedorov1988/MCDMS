import numpy as np
import math
import sys


def calc_force_energy(wave_func, hamiltonian, force_mat, numdims):
    """Calculates forces and energies expectation values
    from wave function and Hamiltonian, force operators"""

    force = np.zeros((numdims))
    for n_dim in range(numdims):
        force[n_dim] = -np.real(expec_value(wave_func, force_mat[n_dim]))

    energy = np.real(expec_value(wave_func, hamiltonian))

    return force, energy


def calc_kin_en(mom, mass, numdims):
    """Calculate kinetic energy of a trajectory"""

    kin_e = sum(0.5 * mom[idim]**2 / mass[idim] for idim in range(numdims))

    return kin_e


def rescale_momentum(pot_e_ini, pot_e_fin, mom_ini, mass, numdims):
    """This subroutine rescales the momentum of the child basis function
    The difference from spawning here is that the average Ehrenfest energy
    is rescaled, not of the pure electronic states"""

    kin_e_ini = calc_kin_en(mom_ini, mass, numdims)
    factor = ((pot_e_ini + kin_e_ini - pot_e_fin) / kin_e_ini)

    if factor < 0.0:
        print "Aborting cloning because because there is\
        not enough energy for momentum adjustment"
        return False, factor

    factor = math.sqrt(factor)
    print "Rescaling momentum by a factor ", factor
    p_fin = factor * mom_ini

    # Computing kinetic energy of child to make sure energy is conserved
    t_fin = 0.0
    for idim in range(numdims):
        t_fin += 0.5 * p_fin[idim] * p_fin[idim] / mass[idim]
    if pot_e_ini + kin_e_ini - pot_e_fin - t_fin > 1e-9:
        print "ENERGY NOT CONSERVED!!!"
        sys.exit

    return True, factor * mom_ini


def symplectic_backprop(H, wf, el_timestep, nsteps, n_krylov, numstates):
    """Immediately after cloning we do not have the electronic wf
    from previous steps since the wf is split into two,
    so we backpropagate it for both parent and child"""

    c_r = np.real(wf)
    c_i = np.imag(wf)
    wf_store = np.zeros((numstates, n_krylov), dtype=np.complex128)
    for n_step in range(nsteps / 2):

        c_r_dot = np.dot(H, c_i)
        c_r = c_r - 0.5 * el_timestep * c_r_dot
        c_i_dot = -1.0 * np.dot(H, c_r)
        c_i = c_i - el_timestep * c_i_dot
        c_r_dot = np.dot(H, c_i)
        c_r = c_r - 0.5 * el_timestep * c_r_dot

        # Storing electronic wf to obtain approximate eigenstates
        wf_store[:, 2*n_step] = c_r
        wf_store[:, 2*n_step + 1] = c_i

    return wf_store.copy()


def propagate_symplectic(self, H, wf, timestep, nsteps, n_krylov):
    """Symplectic split propagator, similar to classical Velocity-Verlet"""

    el_timestep = timestep / nsteps
    c_r = np.real(wf)
    c_i = np.imag(wf)
    n = 0  # counter for how many saved electronic wf components we have
    for i in range(nsteps):

        c_r_dot = np.dot(H, c_i)
        c_r = c_r + 0.5 * el_timestep * c_r_dot
        c_i_dot = -1.0 * np.dot(H, c_r)
        c_i = c_i + el_timestep * c_i_dot
        c_r_dot = np.dot(H, c_i)
        c_r = c_r + 0.5 * el_timestep * c_r_dot

        # This loop saves the electronic wf for transformation to approximate
        # eigenstates
        if nsteps - i <= n_krylov / 2:

            self.wf_store[:, n] = c_r
            self.wf_store[:, n+1] = c_i
            n += 2

    wf = c_r + 1j * c_i

    return wf


def calc_amp_pop(eigenvecs, wave_func, nstates):
    """Calculates amplitudes and population from wave function, eigenvectors"""

    pop = np.zeros(nstates)
    amp = np.zeros((nstates), dtype=np.complex128)
    for j in range(nstates):
        amp[j] = np.dot(eigenvecs[:, j], wave_func)
        pop[j] = np.real(bra_ket(amp[j], amp[j]))

    return amp, pop


def clone_to_a_state(eigenvecs, amp, istate, nstates):
    '''Calculates the child and parent wave functions during
    cloning to a state'''

    child_wf = np.zeros((nstates), dtype=np.complex128)
    parent_wf = np.zeros((nstates), dtype=np.complex128)

    for kstate in range(nstates):
        if kstate == istate:
            # the population is transferred to state i on child
            child_wf += eigenvecs[:, kstate] * amp[kstate]\
                        / np.abs(amp[kstate])

        else:
            # rescaling the rest of the states on the parent function
            parent_wf += eigenvecs[:, kstate] * amp[kstate]\
                       / np.sqrt(1 - bra_ket(amp[istate],
                                             amp[istate]))

    return child_wf, parent_wf


def bra_ket(bra, ket):

    result = np.dot(np.transpose(np.conjugate(bra)), ket)
    return result


def expec_value(vector, operator):
    """Calculates expectation value < bra | operator | ket >"""

    bra = np.transpose(np.conjugate(vector))
    ket = np.dot(operator, vector)
    result = np.dot(bra, ket)

    return result
