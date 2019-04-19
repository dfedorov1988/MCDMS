"""Trajectory objects contain individual trajectory basis functions"""

import sys
import h5py
import numpy as np
from scipy import linalg as lin
from pyspawn.fmsobj import fmsobj
from pyspawn.potential.linear_slope import construct_ham_lin_slope
from pyspawn.misc import calc_amp_pop, clone_to_a_state,\
    bra_ket, expec_value, propagate_symplectic, symplectic_backprop,\
    rescale_momentum, calc_kin_en, calc_force_energy


class traj(fmsobj):
    """Trajectory class. Each instance represents a nuclear basis function"""

    def __init__(self, numdims, numstates, krylov_sub_n):

        self.numdims = numdims
        self.time = 0.0
        self.maxtime = -1.0
        self.mintime = 0.0
        self.firsttime = 0.0
        self.positions = np.zeros(self.numdims)
        self.momenta = np.zeros(self.numdims)
        self.momenta_full_ts = np.zeros(self.numdims)

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

        self.H_elec = np.zeros((self.numstates, self.numstates),
                               dtype=np.complex128)
        self.first_step = False
        self.full_H = bool()
        self.new_amp = np.zeros((1), dtype=np.complex128)
        self.rescale_amp = np.zeros((1), dtype=np.complex128)
        self.n_el_steps = np.zeros((1), dtype=np.int32)
        self.td_wf_full_ts = np.zeros((self.numstates), dtype=np.complex128)
        self.td_wf = np.zeros((self.numstates), dtype=np.complex128)
        self.mce_amps = np.zeros((self.numstates), dtype=np.complex128)
        self.populations = np.zeros(self.numstates)
        self.av_energy = 0.0
        self.av_force = np.zeros(self.numdims)
        self.eigenvecs = np.zeros((self.numstates, self.numstates),
                                  dtype=np.complex128)
        self.approx_eigenvecs = np.zeros((self.krylov_sub_n, self.krylov_sub_n),
                                         dtype=np.complex128)
        self.approx_energies = np.zeros(self.krylov_sub_n)
        self.approx_amp = np.zeros((self.krylov_sub_n), dtype=np.complex128)
        self.approx_pop = np.zeros(self.krylov_sub_n)
        self.approx_wf_full_ts = np.zeros((self.krylov_sub_n),
                                          dtype=np.complex128)
        self.wf_store_full_ts = np.zeros((self.numstates, self.krylov_sub_n),
                                         dtype=np.complex128)
        self.wf_store = np.zeros((self.numstates, self.krylov_sub_n),
                                 dtype=np.complex128)
        self.clone_e_gap = np.zeros(self.numstates)
        self.clone_e_gap_prev = np.zeros(self.numstates)

    def init_traj(self, time, ndims, pos, mom, wid, masses, nstates, istat, lab):
        """Initialize trajectory"""

        self.time = time
        self.positions = pos
        self.momenta = mom
        self.widths = wid
        self.masses = masses
        self.label = lab
        self.numstates = nstates
        self.firsttime = time
        self.numdims = ndims

    def inherit_traj_param(self, parent):
        """Copies parameters from parent basis function"""

        self.timestep = parent.timestep
        self.maxtime = parent.maxtime
        self.full_H = parent.full_H
        self.n_el_steps = parent.n_el_steps
        self.widths = parent.widths
        self.masses = parent.masses
        self.clone_e_gap = np.zeros(parent.clone_e_gap.shape[0])

        if hasattr(parent, 'atoms'):
            self.atoms = parent.atoms
        if hasattr(parent, 'civecs'):
            self.civecs = parent.civecs
            self.ncivecs = parent.ncivecs
        if hasattr(parent, 'orbs'):
            self.orbs = parent.orbs
            self.norbs = parent.norbs
        if hasattr(parent, 'prev_wf_positions'):
            self.prev_wf_positions = parent.prev_wf_positions
        if hasattr(parent, 'electronic_phases'):
            self.electronic_phases = parent.electronic_phases

    def init_clone_traj_approx(self, parent, istate, label, nuc_norm):
        """Initialize cloned trajectory (cloning to a state)
        from approximate eigenstates"""

        self.inherit_traj_param(parent)
        time = parent.time
        self.time = time
        self.label = label

        pos_t = parent.positions
        mom_t = parent.momenta_full_ts
        tmp_energy = parent.av_energy

        H_full, force_full = parent.construct_el_H(pos_t)
        eigenvals, eigenvectors = np.linalg.eigh(H_full)

        amp = parent.approx_amp
        # Transforming full Hamiltonian into Krylov subspace using
        # electronic wf from the previous electronic timesteps
        tr_matrix = np.linalg.qr(parent.wf_store_full_ts)[0]

        Hk = expec_value(tr_matrix, H_full)
        approx_force = np.zeros((self.numdims, self.krylov_sub_n,
                                 self.krylov_sub_n), dtype=np.complex128)
        for n_dim in range(self.numdims):
            approx_force[n_dim] = expec_value(tr_matrix, force_full[n_dim])

        approx_eigenvecs = np.linalg.eigh(Hk)[1]
        # Getting eigenvectors from approximate ones
        eigenvectors = np.dot(
            tr_matrix, np.dot(approx_eigenvecs,
                              np.transpose(np.conjugate(tr_matrix))))
        nstates = self.krylov_sub_n
        eigenvecs = approx_eigenvecs
        H = Hk
        force = approx_force

        child_wf, parent_wf = clone_to_a_state(eigenvecs, amp,
                                               istate, nstates)
        child_amp, child_pop = calc_amp_pop(eigenvecs,
                                            child_wf, nstates)
        parent_amp, parent_pop = calc_amp_pop(eigenvecs,
                                              parent_wf, nstates)
        child_force, child_energy = calc_force_energy(
            child_wf, H, force, self.numdims)
        parent_force, parent_energy = calc_force_energy(
            parent_wf, H, force, parent.numdims)

        print "Rescaling child's momentum:"
        child_rescale_ok, child_rescaled_momenta =\
            rescale_momentum(tmp_energy, child_energy,
                             mom_t, self.masses, self.numdims)

        if child_rescale_ok:

            parent_tot_e = tmp_energy + calc_kin_en(mom_t, parent.masses, parent.numdims)
            child_tot_e = child_energy +\
            calc_kin_en(child_rescaled_momenta, self.masses, self.numdims)
            print "child_E after rescale =", child_tot_e
            print "parent E before rescale=", parent_tot_e
            print "Rescaling parent's momentum"
            parent_rescale_ok, parent_rescaled_momenta\
            = rescale_momentum(tmp_energy, float(parent_energy),
                               mom_t, parent.masses, parent.numdims)
            if not parent_rescale_ok:
                return False
            print "parent E after rescale = ",\
            calc_kin_en(parent_rescaled_momenta, parent.masses, parent.numdims)\
            + parent_energy

            if parent_rescale_ok:
                # Setting mintime to current time to avoid backpropagation
                mintime = parent.time
                self.mintime = mintime
                self.firsttime = time
                self.positions = parent.positions

                # child amplitude rescale
                self.rescale_amp[0] = amp[istate] / child_amp[istate]
                # parent amplitude rescale
                for n  in range(nstates):
                    if n != istate:
                        parent.rescale_amp[0] = amp[n] / parent_amp[n]
                print "Rescaling parent amplitude by a factor ", parent.rescale_amp
                print "Rescaling child amplitude by a factor ", self.rescale_amp

                # Updating quantum parameters of child
                self.av_energy = float(child_energy)
                self.av_force = child_force
                self.first_step = True
                self.momenta = child_rescaled_momenta
                self.momenta_full_ts = child_rescaled_momenta
                self.eigenvecs = eigenvectors
                self.energies = eigenvals

                # Wave function in original basis
                child_wf_orig_basis = np.dot(tr_matrix, child_wf)
                self.td_wf_full_ts = child_wf_orig_basis
                self.td_wf = child_wf_orig_basis
                # Populations and amplitudes in original basis
                child_full_amp, child_full_pop =\
                    calc_amp_pop(eigenvectors, child_wf_orig_basis,
                                 self.numstates)
                self.populations = child_full_pop
                self.mce_amps = child_full_amp
                self.approx_eigenvecs = approx_eigenvecs
                self.approx_amp = child_amp
                self.approx_pop = child_pop

                self.h5_output()

                # Updating quantum parameters of parent
                parent.momenta = parent_rescaled_momenta
                parent.momenta_full_ts = parent_rescaled_momenta
                parent.av_energy = float(parent_energy)
                parent.av_force = parent_force
                parent.energies = eigenvals
                parent.eigenvecs = eigenvectors

                # Transforming approximate wf into original basis
                parent_wf_orig_basis = np.dot(tr_matrix, parent_wf)
                parent.td_wf = parent_wf_orig_basis
                parent.td_wf_full_ts = parent_wf_orig_basis
                # Populations and amplitudes in original basis
#                 parent_full_amp, parent_full_pop =\
#                     calc_amp_pop(eigenvectors, parent_wf_orig_basis,
#                                  self.numstates)
#                     parent.populations = parent_full_pop
#                     parent.mce_amps = parent_full_amp
                parent.approx_amp = parent_amp
                parent.approx_pop = parent_pop
                parent.approx_eigenvecs = approx_eigenvecs

                # this makes sure the parent trajectory in VV propagated
                # as first step because the wave function is at the full TS,
                # should be half step ahead
                parent.first_step = True

                return True

    def init_clone_traj_to_a_state(self, parent, istate, label, nuc_norm):
        """Initialize cloned trajectory (cloning to a state, same way as in original
        paper on AIMC"""

        self.inherit_traj_param(parent)

        time = parent.time
        self.time = time
        self.label = label
        pos_t = parent.positions
        mom_t = parent.momenta_full_ts
        tmp_amp = parent.mce_amps
        tmp_energy = parent.av_energy

        H_full, force = self.construct_el_H(pos_t)
        eigenvals, eigenvectors = lin.eigh(H_full)

        child_wf, parent_wf = clone_to_a_state(
            eigenvectors, tmp_amp, istate, self.numstates)
        child_amp, child_pop = calc_amp_pop(
            eigenvectors, child_wf, self.numstates)
        parent_amp, parent_pop = calc_amp_pop(
            eigenvectors, parent_wf, self.numstates)
        child_force, child_energy = calc_force_energy(
            child_wf, H_full, force, self.numdims)
        parent_force, parent_energy = calc_force_energy(
            parent_wf, H_full, force, parent.numdims)

        print "Rescaling child's momentum:"
        child_rescale_ok, child_rescaled_momenta\
        = rescale_momentum(tmp_energy, child_energy,
                           mom_t, self.masses, self.numdims)

        if child_rescale_ok:

            parent_tot_e = tmp_energy\
                + calc_kin_en(mom_t, parent.masses, parent.numdims)
            child_tot_e = child_energy +\
                calc_kin_en(child_rescaled_momenta, self.masses, parent.numdims)
            print "child_E after rescale =", child_tot_e
            print "parent E before rescale=", parent_tot_e
            print "Rescaling parent's momentum"
            parent_rescale_ok, parent_rescaled_momenta\
            = rescale_momentum(
                tmp_energy, float(parent_energy),
                mom_t, parent.masses, parent.numdims)
            if not parent_rescale_ok:
                return False
            print "parent E after rescale = ",\
            calc_kin_en(parent_rescaled_momenta, parent.masses, parent.numdims)\
                + parent_energy

            if parent_rescale_ok:

                # Setting mintime to current time to avoid backpropagation
                mintime = parent.time
                self.mintime = mintime
                self.firsttime = time
                self.positions = parent.positions

                # child amplitude rescale
                self.rescale_amp[0] = tmp_amp[istate] / child_amp[istate]
                # parent amplitude rescale
                for n_state  in range(self.numstates):
                    if n_state != istate:
                        parent.rescale_amp[0] = tmp_amp[n_state] / parent_amp[n_state]
                print "Rescaling parent amplitude by a factor ", parent.rescale_amp
                print "Rescaling child amplitude by a factor ", self.rescale_amp

                # Updating quantum parameters for child
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
                self.h5_output()

                # Updating quantum parameters for parent
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
                parent.first_step = True # this makes sure the parent trajectory
                # in VV propagated as first step because the wave function is
                # at the full TS, should be half step ahead

                return True

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
        self.clone_e_gap_prev = self.clone_e_gap
        self.clone_e_gap = self.compute_cloning_e_gap_thresh()

    def compute_cloning_e_gap_thresh(self):
        """Computing the energy differences between each state
        and the average"""

        if self.full_H:
            delta_e_array = np.zeros(self.numstates)
            for istate in range(self.numstates):
                delta_e = np.abs(self.energies[istate] - self.av_energy)
                delta_e_array[istate] = delta_e

        if not self.full_H:
            delta_e_array = np.zeros(self.krylov_sub_n)
            for istate in range(self.krylov_sub_n):

                delta_e = np.abs(self.approx_energies[istate] - self.av_energy)
                delta_e_array[istate] = delta_e

        return delta_e_array

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
            length = dset.len()

            """This is not ideal, but couldn't find a better way to do this:
               During cloning the parent ES parameters change,
               but the hdf5 file already has the data for the timestep,
               so this just overwrites the previous values
               if parameter first step is true"""

            if self.first_step and self.time > 1e-6 and length > 0:
                ipos = length - 1
            else:
                dset.resize(length + 1, axis=0)
                ipos = length

            getcom = "self." + key
            tmp = eval(getcom)
            if key == 'wf_store':
                tmp = np.ndarray.flatten(np.real(tmp))

            if n != 1:
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
        if hasattr(self, "atoms"):
            trajgrp.attrs["atoms"] = self.atoms

    def get_data_at_time_from_h5(self, time_req, dset_name):
        """This subroutine gets trajectory data from "dset_name" array at a certain time"""

        h5f = h5py.File("working.hdf5", "r")
        groupname = "traj_" + self.label
        trajgrp = h5f.get(groupname)
        dset_time = trajgrp["time"][:]
        ipoint = -1
        for i, cur_time in enumerate(dset_time):
            if (cur_time < time_req + 1.0e-6) and (cur_time > time_req - 1.0e-6):
                ipoint = i
                #print "dset_time[i] ", dset_time[i]
                #print "i ", i
        dset = trajgrp[dset_name][:]
        data = np.zeros(len(dset[ipoint, :]))
        data = dset[ipoint, :]
        #print "dset[ipoint,:] ", dset[ipoint,:]
        h5f.close()
        return data

    def get_all_qm_data_at_time_from_h5(self, time_req, suffix=""):
        """This subroutine pulls all arrays from the trajectory at a certain time and assigns
        results to the same variable names with _qm suffix. The _qm variables essentially
        match the values at _t, added for clarity."""

        h5f = h5py.File("working.hdf5", "r")
        groupname = "traj_" + self.label
        trajgrp = h5f.get(groupname)
        dset_time = trajgrp["time"][:]
        #print "size", dset_time.size
        ipoint = -1
        for i, cur_time in enumerate(dset_time):
            if (cur_time < time_req + 1.0e-6) and (cur_time > time_req - 1.0e-6):
                ipoint = i

        for dset_name in self.h5_datasets:
            dset = trajgrp[dset_name][:]
            data = np.zeros(len(dset[ipoint, :]))
            data = dset[ipoint, :]
            comm = "self." + dset_name + "_qm" + suffix + " = data"
            exec(comm)
#             print "comm ", comm
#             print "dset[ipoint,:] ", dset[ipoint,:]
        h5f.close()

    def prop_first_step(self):
        """Classical Velocity-Verlet propagator for the first time step"""

        print "Performing the VV propagation for the first timestep"
        dt = self.timestep
        x_t = self.positions
        p_t = self.momenta
        mass = self.masses
        v_t = p_t / mass
        t = self.time

        self.compute_elec_struct()
        self.h5_output()
        self.first_step = False

        f_t = self.av_force
        a_t = f_t / mass

        # propagating velocity half a timestep
        v_tphdt = v_t + 0.5 * a_t * dt

        # now we can propagate position full timestep
        x_tpdt = x_t + v_tphdt * dt
        self.positions = x_tpdt
        self.compute_elec_struct()
        f_tpdt = self.av_force
        a_tpdt = f_tpdt / mass

        # we can compute momentum value at full timestep (t0 + dt)
        v_tpdt = v_tphdt + 0.5 * a_tpdt * dt
        p_tpdt = v_tpdt * mass

        # Positions and momenta values at t0 + dt
        self.momenta = p_tpdt
        t += dt
        self.time = t

        # Output of parameters at t0 + dt
        self.h5_output()

        # Saving momentum at t0 + 3/2 * dt for the next iteration
        v_tp3hdt = v_tpdt + 0.5 * a_tpdt * dt
        p_tp3hdt = v_tp3hdt * mass
        self.momenta = p_tp3hdt
        self.momenta_full_ts = p_tpdt

    def prop_not_first_step(self):
        """Velocity Verlet propagator for not first timestep. Here we call electronic structure
        property calculation only once"""

        dt = self.timestep
        x_t = self.positions
        mass = self.masses

        p_tphdt = self.momenta
        v_tphdt = p_tphdt / mass
        t = self.time

        # Propagating positions to t + dt and computing electronic structure
        x_tpdt = x_t + v_tphdt * dt
        self.positions = x_tpdt
        self.compute_elec_struct()

        force_tpdt = self.av_force
        a_tpdt = force_tpdt / mass
        v_tpdt = v_tphdt + 0.5 * a_tpdt * dt
        p_tpdt = v_tpdt * mass

        self.momenta_full_ts = p_tpdt
        self.momenta = p_tpdt

        t += dt
        self.time = t

        # Output of parameters at t
        self.h5_output()

        # Computing and saving momentum at t + 1/2 dt
        v_tp3hdt = v_tpdt + 0.5 * a_tpdt * dt
        p_tp3hdt = v_tp3hdt * mass
        self.momenta = p_tp3hdt

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

        sigx = np.sqrt(1.0 / (4.0 * alphax))
        sigp = np.sqrt(alphax)

        dtheta = 2.0 * np.pi * np.random.rand(ndims - 6)
        dr = np.sqrt(np.random.rand(ndims - 6))

        dx1 = dr * np.sin(dtheta)
        dx2 = dr * np.cos(dtheta)

        rsq = dx1 * dx1 + dx2 * dx2

        fac = np.sqrt(-2.0 * np.log(rsq) / rsq)

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

            print "Position =", self.positions
            print "Average energy =", self.av_energy
            print "Energies =", ss_energies
            print "approx wf =\n", self.approx_wf_full_ts
            print "approx amp", self.approx_amp
            print "approx_e =", approx_e
            print "approx Population = ", approx_pop
            print "population =", self.populations
            print "norm =", sum(pop)

        n_krylov = self.krylov_sub_n
        wf = self.td_wf

        print "Performing electronic structure calculations:"
        pos = self.positions[0]

        n_el_steps = self.n_el_steps
        el_timestep = self.timestep / n_el_steps

        # Constructing Hamiltonian, computing derivatives,
        # for now solving the eigenvalue problem to get adiabatic states
        # for real systems it will be replaced by will get it from ES program
        H_elec, force = self.construct_el_H(pos)
        ss_energies, eigenvectors = lin.eigh(H_elec)

        pop = np.zeros(self.numstates)
        amp = np.zeros((self.numstates), dtype=np.complex128)
        approx_pop = np.zeros(self.krylov_sub_n)
        approx_amp = np.zeros((self.krylov_sub_n), dtype=np.complex128)

        if np.dot(np.transpose(np.conjugate(wf)), wf) < 1e-8:
            print "WF = 0, constructing electronic wf for the first timestep"
            wf = eigenvectors[:, -1] # starting on the highest energy state
        else:
            if not self.first_step:
                print "\nPropagating electronic wave function not first step"
                wf = propagate_symplectic(self, (H_elec), wf, self.timestep/2,
                                          n_el_steps/2, n_krylov)
                self.wf_store_full_ts = self.wf_store.copy()

            if self.first_step:
                print "\nFirst step, skipping electronic wave function propagation"
                self.wf_store_full_ts = symplectic_backprop(
                    H_elec, wf, el_timestep, n_krylov, n_krylov, self.numstates)

        wf_T = np.transpose(np.conjugate(wf))
        av_energy = np.real(np.dot(np.dot(wf_T, H_elec), wf))

        tr_matrix = np.linalg.qr(self.wf_store_full_ts)[0]
        Hk = np.dot(np.transpose(np.conjugate(tr_matrix)),
                    np.dot(H_elec, tr_matrix))
        Fk = np.dot(np.transpose(np.conjugate(tr_matrix)),
                    np.dot(force, tr_matrix))

        approx_e, approx_eigenvecs = np.linalg.eigh(Hk)
        self.approx_energies = approx_e
        self.approx_eigenvecs = approx_eigenvecs

        av_force = np.zeros((self.numdims))
        for n in range(self.numdims):
            av_force[n] = -np.real(np.dot(np.dot(wf_T, force[n]), wf))

        # test: getting real eigenvectors from approximate
        eigenvectors = np.dot(np.dot(tr_matrix, approx_eigenvecs),
                              np.transpose(np.conjugate(tr_matrix)))
        for j in range(self.numstates):
            amp[j] = np.dot(np.conjugate(np.transpose(eigenvectors[:, j])), wf)
            pop[j] = np.real(np.dot(np.transpose(np.conjugate(amp[j])), amp[j]))

        norm = np.dot(wf_T, wf)
        if abs(norm - 1.0) > 1e-6:
            print "WARNING: Norm is not conserved!!! N =", norm

        approx_wf = np.dot(np.conjugate(np.transpose(tr_matrix)), wf)

        for j in range(self.krylov_sub_n):
            # calculating approximate amplitudes and populations
            approx_amp[j] = np.dot(
                np.conjugate(np.transpose(approx_eigenvecs[:, j])), approx_wf)
            approx_pop[j] = np.real(
                np.dot(np.transpose(np.conjugate(approx_amp[j])), approx_amp[j]))

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
        wf = propagate_symplectic(self, H_elec, wf, self.timestep / 2, n_el_steps / 2, n_krylov)

        # Saving electronic wf and Hamiltonian
        self.td_wf = wf
        self.H_elec = H_elec

    def construct_el_H(self, pos):
        """Constructing n state 1D system and computing d/dx, d/dy for
        force computation.
        Later will be replaced with the electronic structure program call"""

        H_elec, force = construct_ham_lin_slope(pos, self.numstates)

        return H_elec, force

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
