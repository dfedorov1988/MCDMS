# trajectory objects contain individual trajectory basis functions
import numpy as np
import sys
import math
from pyspawn.fmsobj import fmsobj
import os
import shutil
import h5py
from scipy import linalg as lin
from numpy import dtype
from pyspawn.potential.linear_slope import propagate_symplectic
import cmath
from scipy.optimize import fsolve, root, broyden1
from datashape.coretypes import int32


def bra_ket(bra, ket):

    result = np.dot(np.transpose(np.conjugate(bra)), ket)
    return result


def expec_value(wf, operator):
    """Calculates expectation value < bra | operator | ket >"""

    bra = np.transpose(np.conjugate(wf))
    ket = np.dot(operator, wf)
    result = np.dot(bra, ket)

    return result


class traj(fmsobj):
        
    def __init__(self, numdims, numstates, krylov_sub_n):

        self.numdims = numdims        
        self.time = 0.0
        self.maxtime = -1.0
        self.mintime = 0.0
        self.firsttime = 0.0
        self.positions = np.zeros(self.numdims)
        self.momenta = np.zeros(self.numdims)
        self.widths = np.zeros(self.numdims)
        self.masses = np.zeros(self.numdims)
        self.label = "00"
        self.h5_datasets = dict()

        self.timestep = 0.0
        self.numstates = numstates
        self.krylov_sub_n = krylov_sub_n
        
        self.length_wf = self.numstates
        self.wf = np.zeros((self.numstates, self.length_wf))
        self.prev_wf = np.zeros((self.numstates, self.length_wf))
        self.energies = np.zeros(self.numstates)
        self.forces = np.zeros((self.numstates, self.numdims))
        self.S_elec_flat = np.zeros(self.numstates * self.numstates)

        self.numchildren = 0

        self.positions_qm = np.zeros(self.numdims)
        self.momenta_qm = np.zeros(self.numdims)
        self.energies_qm = np.zeros(self.numstates)
        self.forces_i_qm = np.zeros(self.numdims)

        #In the following block there are variables needed for ehrenfest
        self.H_elec = np.zeros((self.numstates, self.numstates),
                               dtype = np.complex128)
        self.first_step = False
        self.full_H = bool()
        self.new_amp = np.zeros((1), dtype = np.complex128)
        self.rescale_amp = np.zeros((1), dtype = np.complex128)
        self.n_el_steps = np.zeros((1), dtype = np.int32)
        self.td_wf_full_ts = np.zeros((self.numstates), dtype = np.complex128)
        self.td_wf = np.zeros((self.numstates), dtype = np.complex128)
        self.mce_amps = np.zeros((self.numstates), dtype = np.complex128)
        self.populations = np.zeros(self.numstates)
        self.av_energy = 0.0
        self.av_force = np.zeros(self.numdims)
        self.eigenvecs = np.zeros((self.numstates, self.numstates),
                                  dtype = np.complex128)
        self.approx_eigenvecs = np.zeros((self.krylov_sub_n, self.krylov_sub_n),
                                         dtype = np.complex128)
        self.approx_energies = np.zeros(self.krylov_sub_n)
        self.approx_amp = np.zeros((self.krylov_sub_n), dtype = np.complex128)
        self.approx_pop = np.zeros(self.krylov_sub_n)
        self.approx_wf_full_ts = np.zeros((self.krylov_sub_n),
                                          dtype = np.complex128)
        self.wf_store_full_ts = np.zeros((self.numstates, self.krylov_sub_n),
                                         dtype = np.complex128)
        self.wf_store = np.zeros((self.numstates, self.krylov_sub_n),
                                 dtype = np.complex128)
        self.clone_E_diff = np.zeros(self.numstates)
        self.clone_E_diff_prev = np.zeros(self.numstates)

    def calc_kin_en(self, p, m):
        """Calculate kinetic energy of a trajectory"""

        ke = sum(0.5 * p[idim]**2 / m[idim] for idim in range(self.numdims))

        return ke

    def init_traj(self, t, ndims, pos, mom, wid, m, nstates, istat, lab):
        """Initialize trajectory"""

        self.time = t
        self.positions = pos
        self.momenta = mom
        self.widths = wid
        self.masses = m
        self.label = lab
        self.numstates = nstates
        self.firsttime = t

    def inherit_traj_param(self, parent):
        """Copies parameters from parent basis function"""

        self.timestep = parent.timestep
        self.maxtime = parent.maxtime
        self.full_H = parent.full_H
        self.n_el_steps = parent.n_el_steps
        self.widths = parent.widths
        self.masses = parent.masses
        self.clone_E_diff = np.zeros(parent.clone_E_diff.shape[0])

        if hasattr(parent,'atoms'):
            self.atoms = parent.atoms
        if hasattr(parent,'civecs'):
            self.civecs = parent.civecs
            self.ncivecs = parent.ncivecs
        if hasattr(parent,'orbs'):
            self.orbs = parent.orbs
            self.norbs = parent.norbs
        if hasattr(parent,'prev_wf_positions'):
            self.prev_wf_positions = parent.prev_wf_positions
        if hasattr(parent,'electronic_phases'):
            self.electronic_phases = parent.electronic_phases
        self.potential_specific_traj_copy(parent)

    def init_clone_traj_approx(self, parent, istate, label, nuc_norm):
        """Initialize cloned trajectory (cloning to a state)
        from approximate eigenstates"""

        self.inherit_traj_param(parent)
        time = parent.time
        self.time = time
        self.label = label
        pos_t = parent.positions
        mom_t = parent.momenta_full_ts
        tmp_pop = parent.approx_pop
        tmp_amp = parent.approx_amp
        tmp_force = parent.av_force
        tmp_energy = parent.av_energy
        eigenvals = parent.energies
        tmp_wf = parent.approx_wf_full_ts

        H_elec, Force = parent.construct_el_H(pos_t)
        eigenvals, eigenvectors = np.linalg.eigh(H_elec)

        # Transforming full Hamiltonian into Krylov subspace using electronic
        # wf from the previous electronic timesteps
        q, r = np.linalg.qr(parent.wf_store_full_ts)

        Hk = expec_value(q, H_elec)
        approx_force = np.zeros((self.numdims, self.krylov_sub_n,
                                 self.krylov_sub_n), dtype=np.complex128)
        for n in range(self.numdims):
            approx_force[n] = expec_value(q, Force[n])

        approx_e, approx_eigenvecs = np.linalg.eigh(Hk)
        child_wf = np.zeros((self.krylov_sub_n), dtype=np.complex128) 
        parent_wf = np.zeros((self.krylov_sub_n), dtype=np.complex128) 

        for kstate in range(self.krylov_sub_n):
            if kstate == istate:
                # the population is transferred to state i on child
                child_wf += approx_eigenvecs[:, kstate] * tmp_amp[kstate]\
                            / np.abs(tmp_amp[kstate])

            else:
                # rescaling the rest of the states on the parent function
                parent_wf += approx_eigenvecs[:, kstate] * tmp_amp[kstate]\
                           / np.sqrt(1 - np.dot(np.conjugate(tmp_amp[istate]),
                                                tmp_amp[istate]))

        child_wf_T = np.conjugate(np.transpose(child_wf))
        parent_wf_T = np.conjugate(np.transpose(parent_wf))
   
        child_pop = np.zeros(self.krylov_sub_n)
        child_amp = np.zeros((self.krylov_sub_n), dtype=np.complex128)
        parent_pop = np.zeros(self.krylov_sub_n)
        parent_amp = np.zeros((self.krylov_sub_n), dtype=np.complex128)     

        for j in range(self.krylov_sub_n):
            child_amp[j] = np.dot(np.conjugate(approx_eigenvecs[:, j]),
                                  child_wf)
            child_pop[j] = np.real(np.dot(np.conjugate(child_amp[j]),
                                          child_amp[j]))
            parent_amp[j] = np.dot(np.conjugate(approx_eigenvecs[:, j]),
                                   parent_wf)
            parent_pop[j] = np.real(np.dot(np.conjugate(parent_amp[j]),
                                           parent_amp[j]))

        parent_force = np.zeros((self.numdims))    
        child_force = np.zeros((self.numdims)) 

        for n in range(self.numdims):
            parent_force[n] = -np.real(expec_value(parent_wf, approx_force[n]))
        for n in range(self.numdims):
            child_force[n] = -np.real(expec_value(child_wf, approx_force[n]))
        child_energy = np.real(expec_value(child_wf, Hk))
        parent_energy = np.real(expec_value(parent_wf, Hk))

        approx_e = expec_value(tmp_wf, Hk)
        exact_e = expec_value(parent.td_wf_full_ts, H_elec)

        print "Rescaling child's momentum:"
        child_rescale_ok, child_rescaled_momenta =\
            self.rescale_momentum(tmp_energy, child_energy, mom_t) 
        
        if child_rescale_ok:

            parent_E_total = tmp_energy + parent.calc_kin_en(mom_t, parent.masses)
            child_E_total = child_energy +\
            self.calc_kin_en(child_rescaled_momenta, self.masses)
            print "child_E after rescale =", child_E_total
            print "parent E before rescale=", parent_E_total 
            print "Rescaling parent's momentum"
            parent_rescale_ok, parent_rescaled_momenta\
            = parent.rescale_momentum(tmp_energy, float(parent_energy), mom_t)
            if not parent_rescale_ok:
                return False
            print "parent E after rescale = ",\
            parent.calc_kin_en(parent_rescaled_momenta, parent.masses)\
            + parent_energy         

            if parent_rescale_ok:
                # Setting mintime to current time to avoid backpropagation
                mintime = parent.time
                self.mintime = mintime
                self.firsttime = time
                self.positions = parent.positions

                # updating quantum parameters for child    
                child_wf_orig_basis = np.dot(q, child_wf)
                self.td_wf_full_ts = child_wf_orig_basis
                self.td_wf = child_wf_orig_basis
                approx_amp = np.zeros((self.numstates), dtype=np.complex128) 
                approx_pop = np.zeros(self.numstates) 
                for j in range(self.numstates):
                    approx_amp[j] = np.dot(np.conjugate(np.transpose(eigenvectors[:, j])),
                                           child_wf_orig_basis)
                    approx_pop[j] = np.real(np.dot(np.transpose(np.conjugate(approx_amp[j])),
                                                   approx_amp[j]))
                print "child_full_pop", approx_pop
                self.av_energy = float(child_energy)
                self.approx_amp = child_amp
                self.approx_pop = child_pop
                self.av_force = child_force
                self.first_step = True
                self.momenta = child_rescaled_momenta
                self.momenta_full_ts = child_rescaled_momenta
                self.eigenvecs = eigenvectors
                self.energies = eigenvals
                self.approx_eigenvecs = approx_eigenvecs
#                 self.approx_energies = float(approx_e)
                # IS THIS OK?!
                self.h5_output()
                
                parent_wf_orig_basis = np.dot(q, parent_wf)
                parent.momenta = parent_rescaled_momenta
                parent.momenta_full_ts = parent_rescaled_momenta
                parent.td_wf = parent_wf_orig_basis
                parent.td_wf_full_ts = parent_wf_orig_basis
                parent.av_energy = float(parent_energy)
                parent.approx_amp = parent_amp
                parent.approx_pop = parent_pop
                parent.av_force = parent_force
                parent.energies = eigenvals
                parent.eigenvecs = eigenvectors
                parent.approx_eigenvecs = approx_eigenvecs
#                 parent.approx_energies = float(approx_e)
                # this makes sure the parent trajectory in VV propagated as first step
                # because the wave function is at the full TS, should be half step ahead
                parent.first_step = True
                print "child_pop =", child_pop
                print "parent_pop =", parent_pop
                self.rescale_amp[0] = np.abs(tmp_amp[istate])
                parent.rescale_amp[0] =\
                    np.sqrt(1 - np.dot(np.conjugate(tmp_amp[istate]),
                    tmp_amp[istate]))
                print "Rescaling parent bf amplitude by a factor ", parent.rescale_amp
                print "Rescaling child bf amplitude by a factor ", self.rescale_amp

                return True
            else:
                return False
            
    def init_clone_traj_to_a_state(self, parent, istate, label, nuc_norm):
        """Initialize cloned trajectory (cloning to a state, same way as in original 
        paper on AIMC"""

        self.inherit_traj_param(parent)
        
        time = parent.time
        self.time = time
        self.label = label
        pos_t = parent.positions
        mom_t = parent.momenta_full_ts
        tmp_pop = parent.populations
        tmp_amp = parent.mce_amps
        tmp_force = parent.av_force
        tmp_energy = parent.av_energy

        tmp_wf = self.td_wf_full_ts
        H_elec, Force = self.construct_el_H(pos_t)
        eigenvals, eigenvectors = lin.eigh(H_elec)
        
        #### Check norm conservation ####
        norm_abk = 0.0
                
        for i in range(self.numstates):
            if i == istate:
                norm_abi = tmp_pop[i]
            else:
                norm_abk += tmp_pop[i]
        
        print "total pop =", sum(tmp_pop)
        print "norm_abi =", norm_abi
        print "norm_abk =", norm_abk        
        #################################
        
        child_wf = np.zeros((self.numstates), dtype=np.complex128) 
        parent_wf = np.zeros((self.numstates), dtype=np.complex128) 
        
        for kstate in range(self.numstates):
            if kstate == istate:
                child_wf += eigenvectors[:, kstate] * tmp_amp[kstate]\
                         / np.abs(tmp_amp[kstate])
               
            else:
                parent_wf += eigenvectors[:, kstate] * tmp_amp[kstate]\
                          / np.sqrt(1 - np.abs(tmp_amp[istate])**2)
        
        child_wf_T = np.conjugate(np.transpose(child_wf))
        parent_wf_T = np.conjugate(np.transpose(parent_wf))
                       
        child_pop = np.zeros(self.numstates)
        child_amp = np.zeros((self.numstates), dtype=np.complex128) 
        parent_pop = np.zeros(self.numstates)
        parent_amp = np.zeros((self.numstates), dtype=np.complex128)         

        for j in range(self.numstates):
            child_amp[j] = np.dot(eigenvectors[:, j], child_wf)
            child_pop[j] = np.real(np.dot(np.conjugate(child_amp[j]),
                                          child_amp[j]))
            parent_amp[j] = np.dot(eigenvectors[:, j], parent_wf)
            parent_pop[j] = np.real(np.dot(np.conjugate(parent_amp[j]),
                                           parent_amp[j]))
        
        parent_force = np.zeros((self.numdims))    
        child_force = np.zeros((self.numdims)) 
        print self.numdims
        for n in range(self.numdims):
            parent_force[n] = -np.real(np.dot(np.dot(parent_wf_T, Force[n]), parent_wf))
        for n in range(self.numdims):
            child_force[n] = -np.real(np.dot(np.dot(child_wf_T, Force[n]), child_wf))
        
        child_energy = np.real(np.dot(np.dot(np.transpose(np.conjugate(child_wf)), H_elec), child_wf))
        parent_energy = np.real(np.dot(np.dot(np.transpose(np.conjugate(parent_wf)),\
                                              H_elec), parent_wf))
        print "Child E =", child_energy
        print "Parent E =", parent_energy        
        print "Rescaling child's momentum:"
        child_rescale_ok, child_rescaled_momenta\
        = self.rescale_momentum(tmp_energy, child_energy, mom_t) 

        if child_rescale_ok:

            parent_E_total = tmp_energy + parent.calc_kin_en(mom_t, parent.masses)
            child_E_total = child_energy +\
            self.calc_kin_en(child_rescaled_momenta, self.masses)
            print "child_E after rescale =", child_E_total
            print "parent E before rescale=", parent_E_total 
            print "Rescaling parent's momentum"
            parent_rescale_ok, parent_rescaled_momenta\
            = parent.rescale_momentum(tmp_energy, float(parent_energy), mom_t)
            if not parent_rescale_ok:
                return False
            print "parent E after rescale = ",\
            parent.calc_kin_en(parent_rescaled_momenta, parent.masses) + parent_energy         

            if parent_rescale_ok:
                # Setting mintime to current time to avoid backpropagation
                mintime = parent.time
                self.mintime = mintime
                self.firsttime = time
                self.positions = parent.positions
    
                # updating quantum parameters for child    
                self.td_wf_full_ts = child_wf
                self.td_wf = child_wf
                self.av_energy = float(child_energy)
                self.mce_amps = child_amp
                self.populations = child_pop
                self.av_force = child_force
                self.first_step = True
                self.momenta = child_rescaled_momenta
                self.momenta_full_ts = child_rescaled_momenta
                self.eigenvecs = eigenvectors
                self.energies = eigenvals
                # IS THIS OK?!
                self.h5_output()
                
                parent.momenta = parent_rescaled_momenta
                parent.momenta_full_ts = parent_rescaled_momenta
                parent.td_wf = parent_wf
                parent.td_wf_full_ts = parent_wf
                parent.av_energy = float(parent_energy)
                parent.mce_amps = parent_amp
                parent.populations = parent_pop
                parent.av_force = parent_force
                parent.energies = eigenvals
                parent.eigenvecs = eigenvectors
                
                # this makes sure the parent trajectory in VV propagated as first step
                # because the wave function is at the full TS, should be half step ahead
                parent.first_step = True
#                 print "AMP_i coeff=", np.abs(tmp_amp[istate])
#                 print "child_pop =", child_pop
#                 print "parent_pop =", parent_pop
#                 print "child_amp =", child_amp
#                 print "parent_amp =", parent_amp
#                 print "tmp_amp = ", tmp_amp
                self.rescale_amp[0] = tmp_amp[istate] / child_amp[istate]
#                 self.rescale_amp[0] = np.abs(tmp_amp[istate])
#                 parent.rescale_amp[0] = tmp_amp[istate] / np.sqrt(1 - np.abs(tmp_amp[istate])**2)
#                 parent.rescale_amp[0] = np.sqrt(1 - np.abs(tmp_amp[istate])**2)
                n = 0
                for amp in parent_amp:
                    if n != istate:
                        parent.rescale_amp[0] = tmp_amp[n] / parent_amp[n] 
                    n += 1
                print "Rescaling parent amplitude by a factor ",
                parent.rescale_amp
                print "Rescaling child amplitude by a factor ",
                self.rescale_amp

                return True
            else:
                return False

    def rescale_momentum(self, v_ini, v_fin, p_ini):
        """This subroutine rescales the momentum of the child basis function
        The difference from spawning here is that the average Ehrenfest energy
        is rescaled, not of the pure electronic states"""

        m = self.masses
        t_ini = self.calc_kin_en(p_ini, m)
        factor = ((v_ini + t_ini - v_fin) / t_ini)

        if factor < 0.0:
            print "Aborting cloning because because there is\
            not enough energy for momentum adjustment"
            return False, factor

        factor = math.sqrt(factor)
        print "Rescaling momentum by a factor ", factor
        p_fin = factor * p_ini
 
        # Computing kinetic energy of child to make sure energy is conserved
        t_fin = 0.0
        for idim in range(self.numdims):
            t_fin += 0.5 * p_fin[idim] * p_fin[idim] / m[idim]
        if v_ini + t_ini - v_fin - t_fin > 1e-9: 
            print "ENERGY NOT CONSERVED!!!"
            sys.exit
        return True, factor * p_ini

    def propagate_step(self):
        """When cloning happens we start a parent wf
        as a new trajectory because electronic structure properties changed,
        first_step variable here ensures that
        we don't write to h5 file twice! (see vv.py)"""

        if float(self.time) - float(self.firsttime) < (self.timestep -1e-6)\
        or self.first_step:
            self.prop_first_step()
        else:
            self.prop_not_first_step()

        # computing energy differences
        self.clone_E_diff_prev = self.clone_E_diff
        self.clone_E_diff = self.compute_cloning_E_diff()
        
    def compute_cloning_E_diff(self):
        """Computing the energy differences between each state
        and the average"""

        if self.full_H:
            clone_dE = np.zeros(self.numstates)
            for istate in range(self.numstates):
                dE = np.abs(self.energies[istate] - self.av_energy)
                clone_dE[istate] = dE

        if not self.full_H:
            clone_dE = np.zeros(self.krylov_sub_n)
            for istate in range(self.krylov_sub_n):

                dE = np.abs(self.approx_energies[istate] - self.av_energy)
                clone_dE[istate] = dE

        return clone_dE

    def h5_output(self):
        """This subroutine outputs all datasets into an hdf5 file
        at each timestep"""

        if len(self.h5_datasets) == 0:
            self.init_h5_datasets()
        filename = "working.hdf5"

        h5f = h5py.File(filename, "a")
        groupname = "traj_" + self.label
        if groupname not in h5f.keys():
            self.create_h5_traj(h5f, groupname)
        trajgrp = h5f.get(groupname)
        all_datasets = self.h5_datasets.copy()
        dset_time = trajgrp["time"][:]
        
        for key in all_datasets:
            n = all_datasets[key]
            dset = trajgrp.get(key)
            l = dset.len()
            
            """This is not ideal, but couldn't find a better way to do this:
               During cloning the parent ES parameters change,
               but the hdf5 file already has the data for the timestep,
               so this just overwrites the previous values 
               if parameter first step is true"""
            
            if self.first_step and self.time > 1e-6 and l>0: 
                ipos = l - 1
            else:    
                dset.resize(l+1, axis=0)
                ipos = l

            getcom = "self." + key 
            tmp = eval(getcom)
            if key == 'wf_store':
                tmp = np.ndarray.flatten(np.real(tmp))
            
            if n!=1:
                dset[ipos, 0:n] = tmp[0:n]
            else:
                dset[ipos, 0] = tmp
        
        h5f.flush()
        h5f.close()

    def create_h5_traj(self, h5f, groupname):
        """create a new trajectory group in hdf5 output file"""

        trajgrp = h5f.create_group(groupname)
        for key in self.h5_datasets:
            n = self.h5_datasets[key]
            if key == "td_wf" or key == "mce_amps" or key == "td_wf_full_ts"\
            or key == "approx_wf_full_ts":
                dset = trajgrp.create_dataset(key, (0, n), maxshape=(None, n),
                                              dtype="complex128")
            else:
                dset = trajgrp.create_dataset(key, (0, n), maxshape=(None, n),
                                               dtype="float64")

        # add some metadata
        trajgrp.attrs["masses"] = self.masses
        trajgrp.attrs["widths"] = self.widths
        trajgrp.attrs["full_H"] = self.full_H
        trajgrp.attrs["krylov_sub_n"] = self.krylov_sub_n
        if hasattr(self,"atoms"):
            trajgrp.attrs["atoms"] = self.atoms

    def get_data_at_time_from_h5(self, t, dset_name):
        """This subroutine gets trajectory data from "dset_name" array at a certain time"""

        h5f = h5py.File("working.hdf5", "r")
        groupname = "traj_" + self.label
        filename = "working.hdf5"
        trajgrp = h5f.get(groupname)
        dset_time = trajgrp["time"][:]
        ipoint = -1
        for i in range(len(dset_time)):
            if (dset_time[i] < t + 1.0e-6) and (dset_time[i] > t - 1.0e-6):
                ipoint = i
                #print "dset_time[i] ", dset_time[i]
                #print "i ", i
        dset = trajgrp[dset_name][:]            
        data = np.zeros(len(dset[ipoint, :]))
        data = dset[ipoint, :]
        #print "dset[ipoint,:] ", dset[ipoint,:]        
        h5f.close()
        return data

    def get_all_qm_data_at_time_from_h5(self, t, suffix=""):
        """This subroutine pulls all arrays from the trajectory at a certain time and assigns
        results to the same variable names with _qm suffix. The _qm variables essentially
        match the values at _t, added for clarity."""

        h5f = h5py.File("working.hdf5", "r")
        groupname = "traj_" + self.label
        filename = "working.hdf5"
        trajgrp = h5f.get(groupname)
        dset_time = trajgrp["time"][:]
        #print "size", dset_time.size
        ipoint = -1
        for i in range(len(dset_time)):
            if (dset_time[i] < t+1.0e-6) and (dset_time[i] > t-1.0e-6):
                ipoint = i
                #print "dset_time[i] ", dset_time[i]
                #print "i ", i
        for dset_name in self.h5_datasets:
            dset = trajgrp[dset_name][:]            
            data = np.zeros(len(dset[ipoint, :]))
            data = dset[ipoint, :]
            comm = "self." + dset_name + "_qm" + suffix + " = data"
            exec(comm)
#             print "comm ", comm
#             print "dset[ipoint,:] ", dset[ipoint,:]        
        h5f.close()

    def initial_wigner(self, iseed):
        print "Randomly selecting Wigner initial conditions"
        ndims = self.numdims

        h5f = h5py.File('hessian.hdf5', 'r')
        
        pos = h5f['geometry'][:].flatten()

        h = h5f['hessian'][:]

        m = self.masses

        sqrtm = np.sqrt(m)

        #build mass weighted hessian
        h_mw = np.zeros_like(h)

        for idim in range(ndims):
            h_mw[idim, :] = h[idim, :] / sqrtm

        for idim in range(ndims):
            h_mw[:, idim] = h_mw[:, idim] / sqrtm

        # symmetrize mass weighted hessian
        h_mw = 0.5 * (h_mw + h_mw.T)

        # diagonalize mass weighted hessian
        evals, modes = np.linalg.eig(h_mw)

        # sort eigenvectors
        idx = evals.argsort()[::-1]
        evals = evals[idx]
        modes = modes[:, idx]

        print 'Eigenvalues of the mass-weighted hessian are (a.u.)'
        print evals
        
        # seed random number generator
        np.random.seed(iseed)
        
        alphax = np.sqrt(evals[0:ndims-6]) / 2.0

        sigx = np.sqrt(1.0 / ( 4.0 * alphax))
        sigp = np.sqrt(alphax)

        dtheta = 2.0 * np.pi * np.random.rand(ndims - 6)
        dr = np.sqrt( np.random.rand(ndims - 6) )

        dx1 = dr * np.sin(dtheta)
        dx2 = dr * np.cos(dtheta)

        rsq = dx1 * dx1 + dx2 * dx2

        fac = np.sqrt( -2.0 * np.log(rsq) / rsq )

        x1 = dx1 * fac
        x2 = dx2 * fac

        posvec = np.append(sigx * x1, np.zeros(6))
        momvec = np.append(sigp * x2, np.zeros(6))

        deltaq = np.matmul(modes, posvec) / sqrtm
        pos += deltaq
        mom = np.matmul(modes, momvec) * sqrtm

        self.positions = pos
        self.momenta = mom

        zpe = np.sum(alphax[0 : (ndims - 6)])
        ke = 0.5 * np.sum(mom * mom / m)

        print "ZPE = ", zpe
        print "Kinetic energy = ", ke

    def overlap_nuc_1d(self, xi, xj, di, dj, xwi, xwj):
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

    def overlap_nuc(self, pos_i, pos_j, mom_i, mom_j, widths_i, widths_j):
        
        Sij = 1.0
        for idim in range(self.numdims):
            xi = pos_i[idim]
            xj = pos_j[idim]
            di = mom_i[idim]
            dj = mom_j[idim]
            xwi = widths_i[idim]
            xwj = widths_j[idim]
            Sij *= self.overlap_nuc_1d(xi, xj, di, dj, xwi, xwj)

        return Sij
