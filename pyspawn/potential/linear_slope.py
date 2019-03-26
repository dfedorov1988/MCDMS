import math
import sys
import numpy as np
from scipy import linalg as lin
from Cython.Compiler.PyrexTypes import c_ref_type

#################################################
### electronic structure routines go here #######
#################################################

# Each electronic structure method requires at least two routines:
#    1) compute_elec_struct_, which computes energies, forces, and wfs
#    2) init_h5_datasets_, which defines the datasets to be output to hdf5
#    3) potential_specific_traj_copy, which copies data that is potential specific 
#    from one traj data structure to another 
#    other ancillary routines may be included as well


def compute_elec_struct(self):
    """Electronic structure model for a 1d system with multiple parallel states 
    that intersect a single state with a different slope 
    This subroutine solves the electronic SE and propagates electronic 
    ehrenfest wave function one nuclear time step that's split into 
    smaller electronic timesteps.
    The propagation using approximate eigenstates is also coded here"""

    def print_stuff():
        """Prints variables for debugging"""

        print 'approx_total_pop = ', approx_total_pop
        print 'approx_total_e = ', approx_total_e
        print "Position =", self.positions
#         print "Hamiltonian =\n", H_elec
#         print "positions =", self.positions
#         print "momentum =", self.momenta
        print "Average energy =", self.av_energy
#         print "wf_store =", self.wf_store_full_ts
        print "Energies =", ss_energies
#         print "Force =", av_force
#         print "Wave function =\n", wf
        print "approx wf =\n", self.approx_wf_full_ts
        print "approx amp", self.approx_amp
        print "approx_e =", approx_e
#         print "approx Eigenvectors =\n", self.approx_eigenvecs
        print "approx Population = ", approx_pop
        print "population =", self.populations 
        print "norm =", sum(pop)
#         print "amps =", amp

    n_krylov = self.krylov_sub_n

    wf = self.td_wf
    print "Performing electronic structure calculations:"
    x = self.positions[0]

    n_el_steps = self.n_el_steps
    time = self.time
    el_timestep = self.timestep / n_el_steps

    # Constructing Hamiltonian, computing derivatives, for now solving the eigenvalue problem
    # to get adiabatic states
    # for real systems it will be replaced by approximate eigenstates
    H_elec, Force = self.construct_el_H(x) 
    ss_energies, eigenvectors = lin.eigh(H_elec)
    eigenvectors_T = np.transpose(np.conjugate(eigenvectors))
    
    pop = np.zeros(self.numstates)
    amp = np.zeros((self.numstates), dtype=np.complex128) 
    approx_pop = np.zeros(self.krylov_sub_n)
    approx_amp = np.zeros((self.krylov_sub_n), dtype=np.complex128)

    if np.dot(np.transpose(np.conjugate(wf)), wf)  < 1e-8:
        print "WF = 0, constructing electronic wf for the first timestep", wf
        wf = eigenvectors[:, -1] # starting on the highest energy state
    else:
        if not self.first_step:
            print "\nPropagating electronic wave function not first step"
            wf = propagate_symplectic(self, (H_elec), wf, self.timestep/2,
                                      n_el_steps/2, n_krylov)
            self.wf_store_full_ts = self.wf_store.copy()

        if self.first_step:
            print "\nFirst step, skipping electronic wave function propagation"
            symplectic_backprop(self, H_elec, wf, el_timestep, n_krylov, n_krylov)

    wf_T = np.transpose(np.conjugate(wf))
    av_energy = np.real(np.dot(np.dot(wf_T, H_elec), wf))    

    q, r = np.linalg.qr(self.wf_store_full_ts)
    Hk = np.dot(np.transpose(np.conjugate(q)), np.dot(H_elec, q))
    Fk = np.dot(np.transpose(np.conjugate(q)), np.dot(Force, q))

    approx_e, approx_eigenvecs = np.linalg.eigh(Hk)
    self.approx_energies = approx_e
    self.approx_eigenvecs = approx_eigenvecs

    av_force = np.zeros((self.numdims))    
    for n in range(self.numdims):
        av_force[n] = -np.real(np.dot(np.dot(wf_T, Force[n]), wf))

    for j in range(self.numstates):
        amp[j] = np.dot(np.conjugate(np.transpose(eigenvectors[:, j])), wf)
        pop[j] = np.real(np.dot(np.transpose(np.conjugate(amp[j])), amp[j]))

    norm = np.dot(wf_T, wf)
    if abs(norm - 1.0) > 1e-6:
        print "WARNING: Norm is not conserved!!! N =", norm

    approx_wf = np.dot(np.conjugate(np.transpose(q)), wf)

    for j in range(self.krylov_sub_n):
        # calculating approximate amplitudes and populations
        approx_amp[j] = np.dot(np.conjugate(np.transpose(approx_eigenvecs[:, j])), approx_wf)
        approx_pop[j] = np.real(np.dot(np.transpose(np.conjugate(approx_amp[j])), approx_amp[j]))

    # Assigning variables to the current trajectory object
    self.approx_amp = approx_amp
    self.approx_pop = approx_pop    
    self.approx_wf_full_ts = np.complex128(approx_wf)
    self.av_energy = float(av_energy)
    self.energies = np.real(ss_energies)
    self.av_force = av_force
    self.eigenvecs = eigenvectors
    self.mce_amps_prev = self.mce_amps
    self.mce_amps = amp
    self.td_wf_full_ts = np.complex128(wf)
    self.populations = pop
    
    # To test single passage only
    if self.momenta[0] < 0:
        print "The trajectory reached the inflection point: exiting"
        sys.exit() 

    #print_stuff()
    
    # This part performs the propagation of the electronic wave function 
    # for ehrenfest dynamics at a half step and save it
    wf = propagate_symplectic(self, H_elec, wf, self.timestep / 2,
                              n_el_steps / 2, n_krylov)

    # Saving electronic wf and Hamiltonian
    self.td_wf = wf
    self.H_elec = H_elec


def construct_el_H(self, x):
    """Constructing n state 1D system and computing d/dx, d/dy for
    force computation.
    Later will be replaced with the electronic structure program call"""

    k = 0.005 # off-diagonal coupling matrix elements
    w1 = 0.25 # slope 1
    w2 = 0.025 # slope 2
    delta = 0.01 # gap between diabatic states

    H_elec = np.zeros((self.numstates, self.numstates))
    Hx = np.zeros((self.numstates, self.numstates))
    H_elec[0, 0] = w1 * (-x)
    Hx[0, 0] = -w1

    for n in range(self.numstates - 1):
        if n < 4: # this if statement adds a gap between middle states
            H_elec[n + 1, n + 1] = w2*x - n*delta 
            if n != 0:
                H_elec[0, n + 1] = k
                H_elec[n + 1, 0] = k
            else:
                H_elec[0, n + 1] = k # / 5
                H_elec[n + 1, 0] = k # / 5
            Hx[n + 1, n + 1] = w2

        else:
            H_elec[n + 1, n + 1] = w2*x - n*delta - 0.08 
            if n != 6 and n != 7 and n != 5:
                H_elec[0, n+1] = k
                H_elec[n+1, 0] = k
            else:
                H_elec[0, n + 1] = 0.0 # k / 5
                H_elec[n + 1, 0] = 0.0 # k / 5
            Hx[n + 1, n + 1] = w2

    Force = [Hx]    

    return H_elec, Force


# def construct_el_H(self, x):
#     """Constructing 2 state 1D system and computing d/dx, d/dy for
#     force computation. 
#     Later will be replaced with the electronic structure program call"""
#      
#     # 2 state simplest system 
#     k = 0.005 # off-diagonal coupling matrix elements
#     w1 = 0.25 # slope 1
#     w2 = 0.025 # slope 2
#  
#     H_elec = np.zeros((self.numstates, self.numstates))
#     Hx = np.zeros((self.numstates, self.numstates))
#     H_elec[0, 0] = w1 * (-x)
#     Hx[0, 0] = -w1
#  
#     H_elec[1, 1] = w2 * x 
#     H_elec[0, 1] = k
#     H_elec[1, 0] = k 
#     Hx[1, 1] = w2
#                
#     Force = [Hx]    
#      
#     return H_elec, Force


def propagate_symplectic(self, H, wf, timestep, nsteps, n_krylov):
    """Symplectic split propagator, similar to classical Velocity-Verlet"""

    el_timestep = timestep / nsteps
    c_r = np.real(wf)
    c_i = np.imag(wf)
    n = 0 # counter for how many saved electronic wf components we have
    for i in range(nsteps):
        
        c_r_dot = np.dot(H, c_i)
        c_r = c_r + 0.5 * el_timestep * c_r_dot
        c_i_dot = -1.0 * np.dot(H, c_r)
        c_i = c_i + el_timestep * c_i_dot  
        c_r_dot = np.dot(H, c_i)
        c_r = c_r + 0.5 * el_timestep * c_r_dot  
        
        if nsteps - i <= n_krylov / 2:
#             self.wf_store[:, n] = c_r + 1j * c_i 
#             n += 1
            self.wf_store[:, n] = c_r
            self.wf_store[:, n+1] = c_i
            n += 2

    wf = c_r + 1j * c_i

    return wf


def symplectic_backprop(self, H, wf, el_timestep, nsteps, n_krylov):
    """Immediately after cloning we do not have the electronic wf 
    from previous steps since the wf is split into two,
    so we backpropagate it for both parent and child"""

    c_r = np.real(wf)
    c_i = np.imag(wf)
    self.wf_store_full_ts = np.zeros((self.numstates, self.krylov_sub_n),
                                     dtype = np.complex128)
    for n in range(nsteps / 2):

        c_r_dot = np.dot(H, c_i)
        c_r = c_r - 0.5 * el_timestep * c_r_dot
        c_i_dot = -1.0 * np.dot(H, c_r)
        c_i = c_i - el_timestep * c_i_dot  
        c_r_dot = np.dot(H, c_i)
        c_r = c_r - 0.5 * el_timestep * c_r_dot  

        # Storing electronic wf to obtain approximate eigenstates
        self.wf_store_full_ts[:, 2*n] = c_r
        self.wf_store_full_ts[:, 2*n + 1] = c_i

    self.wf_store = self.wf_store_full_ts.copy()

    return 


def init_h5_datasets(self):
    """Initializing h5 datasets for the variables that we want to store
    in hdf5 file and use for analysis later"""

    self.h5_datasets["av_energy"] = 1
    self.h5_datasets["av_force"] = self.numdims
    self.h5_datasets["td_wf"] = self.numstates
    self.h5_datasets["mce_amps"] = self.numstates    
    self.h5_datasets["time"] = 1
    self.h5_datasets["energies"] = self.numstates
    self.h5_datasets["positions"] = self.numdims
    self.h5_datasets["momenta"] = self.numdims
    self.h5_datasets["populations"] = self.numstates
    self.h5_datasets["td_wf_full_ts"] = self.numstates
    self.h5_datasets["approx_energies"] = self.krylov_sub_n
    self.h5_datasets['approx_pop'] = self.krylov_sub_n
    self.h5_datasets['wf_store'] = self.krylov_sub_n * self.numstates
    self.h5_datasets['approx_wf_full_ts'] = self.krylov_sub_n

def potential_specific_traj_copy(self,from_traj):
    return
