# simulation object contains the current state of the simulation.
# It is analogous to the "bundle" object in the original FMS code.
import os
import shutil
import time
import copy
import types
import numpy as np
import h5py
import numpy.linalg as la

from pyspawn.fmsobj import fmsobj
from pyspawn.traj import traj
import pyspawn.general as gen
import pyspawn.complexgaussian as cg


class Simulation(fmsobj):
    """This is the main simulation module"""

    def __init__(self, numstates):
        # traj is a dictionary of trajectory basis functions (TBFs)
        self.traj = dict()

        # queue is a list of tasks to be run
        self.queue = ["END"]
        # tasktimes is a list of the simulation times associated with each task
        self.tasktimes = [1e10]

        # olapmax is the maximum overlap allowed for a spawn.  Above this,
        # the spawn is cancelled
        self.olapmax = 0.5

        # Number of electronic states (this needs to be fixed since this
        # is already in traj object)
        self.num_el_states = numstates

        self.pop_threshold = 0.1
        self.e_gap_thresh = 0.0
        self.nuc_pop_thresh = 0.05
        # Type of cloning: either 'toastate' or 'pairwise'
        # 'pairwise' version seems to be not a good thing to do
        self.cloning_type = "toastate"

        # quantum time is the current time of the quantum amplitudes
        self.quantum_time = 0.0

        # timestep for quantum propagation
        self.timestep = 0.0
        self.clone_again = False

        # quantum propagator
        # self.qm_propagator = "RK2"
        # quantum hamiltonian
        # self.qm_hamiltonian = "adiabatic"

        # maps trajectories to matrix element indices
        # (the order of trajectories in the dictionary
        # is not the same as amplitudes)
        self.traj_map = dict()

        # quantum amplitudes
        self.qm_amplitudes = np.zeros(0, dtype=np.complex128)

        # Total electronic population on each electronic states
        # takes into account all nuclear basis functions
        self.el_pop = np.zeros(self.num_el_states)

        # energy shift for quantum propagation (better accuracy
        # if energy is close to 0)
        self.qm_energy_shift = 0.0

        # variables to be output to hdf5 mapped to the size of each data point
        self.h5_datasets = dict()
        self.h5_types = dict()

        # maximium walltime in seconds
        self.max_quantum_time = -1.0

        # maximium walltime in seconds
        self.max_walltime = -1.0

    def set_maxtime_all(self, maxtime):
        """Sets the maximum time for all trajectories"""

        self.max_quantum_time = maxtime
        h = self.timestep
        for key in self.traj:
            self.traj[key].maxtime = maxtime + h

    def add_traj(self, traj):
        """Add a trajectory to the simulation"""

        key = traj.label
        print "Trajectory added:", key
        mintime = traj.mintime
        index = -1
        for key2 in self.traj:
            if mintime < self.traj[key2].mintime:
                if index < 0:
                    index = self.traj_map[key2]
                self.traj_map[key2] += 1
        if index < 0:
            index = len(self.traj)
        self.traj[key] = traj
        self.traj_map[key] = index

    def propagate(self):
        """This is the main propagation loop for the simulation"""

        gen.print_splash()
        t0 = time.clock()
        while True:

            # update the queue (list of tasks to be computed)
            print "\nUpdating task queue"
            self.update_queue()

            # if the queue is empty, we're done!
            print "Time =", self.quantum_time
            print "Checking if we are at the end of the simulation"
#             if (self.queue[0] == "END"):
            if self.quantum_time + 1.0e-6 > self.max_quantum_time:
                print "Propagate DONE, simulation ended gracefully!"
                return

            # end simulation if walltime has expired
            print "Checking if maximum wall time is reached"
            if self.max_walltime < time.time() and self.max_walltime > 0:
                print "Wall time expired, simulation ended gracefully!"
                return

            # it is possible for the queue to run empty but
            # for the job not to be done
            if self.queue[0] != "END":
                # Right now we just run a single task per cycle,
                # but we could parallelize here and send multiple tasks
                # out for simultaneous processing.
                current = self.pop_task()
                print "\nStarting " + current
                eval(current)
                print "Done with " + current
            else:
                print "Task queue is empty"

            # propagate quantum variables if possible
            print "\nPropagating quantum amplitudes if we have enough",\
                "information to do so"

            self.propagate_quantum_as_necessary()

            cond_num = np.linalg.cond(self.S)
            if cond_num > 1000:
                print "BAD S matrix: condition number =", cond_num, "\nExiting"
                return

            # print restart output - this must be the last line in this loop!
            print "Updating restart output"
            self.restart_output()
            print "Elapsed wall time: %6.1f" % (time.clock() - t0)

    def propagate_quantum_as_necessary(self):
        """here we will propagate the quantum amplitudes if we have
        the necessary information to do so.
        we have to determine what the maximum time is for which
        we have all the necessary information to propagate the amplitudes"""

        max_info_time = 1.0e10
        # first check trajectories
        for key in self.traj:

            time = self.traj[key].time
            if time < max_info_time:
                max_info_time = time

        print "We have enough information to propagate to time ", max_info_time

        # now, if we have the necessary info, we propagate
        while max_info_time > (self.quantum_time + 1.0e-6):

            if self.quantum_time > 1.0e-6:
                print "Propagating quantum amplitudes at time",\
                    self.quantum_time
                self.qm_propagate_step()
            else:
                print "Propagating quantum amplitudes at time",\
                    self.quantum_time
                self.qm_propagate_step(zoutput_first_step=True)

            print "\nOutputing quantum information to hdf5"
            self.h5_output()
            self.calc_approx_el_populations()
            print "\nNow we will clone new trajectories if necessary:"

            if self.cloning_type == "toastate":
                self.clone_to_a_state()
                if self.clone_again:
                    for key in self.traj:
                        self.traj[key].compute_cloning_e_gap_thresh()
                    self.clone_to_a_state()

    def qm_propagate_step(self, zoutput_first_step=False):
        """Exponential integrator (***Needs reference***)"""

        c1i = (complex(0.0, 1.0))
        self.compute_num_traj_qm()
        qm_t = self.quantum_time
        dt = self.timestep
        qm_tpdt = qm_t + dt

        amps_t = self.qm_amplitudes
        print "Building effective Hamiltonian for the first half step"
        self.build_Heff_half_timestep()
        self.calc_approx_el_populations()
#         norm = np.dot(np.conjugate(np.transpose(amps_t)),
#                       np.dot(self.S, amps_t))
#         print "Norm first half =", norm

        # output the first step before propagating
        if zoutput_first_step:
            self.h5_output()

        iHdt = (-0.5 * dt * c1i) * self.Heff
        W, R = la.eig(iHdt)
        X = np.exp(W)
        amps = amps_t
        tmp1 = la.solve(R, amps)
        tmp2 = X * tmp1  # element-wise multiplication
        amps = np.matmul(R, tmp2)

        self.quantum_time = qm_tpdt
        print "Building effective Hamiltonian for the second half step"
        self.build_Heff_half_timestep()
        print "Effective Hamiltonian built"

        iHdt = (-0.5 * dt * c1i) * self.Heff
        W, R = la.eig(iHdt)
        X = np.exp(W)
        tmp1 = la.solve(R, amps)
        tmp2 = X * tmp1  # element-wise multiplication
        amps = np.matmul(R, tmp2)

        self.qm_amplitudes = amps
        norm = np.dot(np.conjugate(np.transpose(amps)),
                      np.dot(self.S, amps))
#         print "Norm second half =", norm
        if abs(norm - 1.0) > 0.01:
            print "Warning: nuclear norm deviated from 1: norm =", norm
        print "Done with quantum propagation"

    def init_amplitudes_one(self):
        """Sets the first amplitude to 1.0 and all others to zero"""

        self.compute_num_traj_qm()
        self.qm_amplitudes = np.zeros_like(self.qm_amplitudes,
                                           dtype=np.complex128)
        self.qm_amplitudes[0] = 1.0

    def compute_num_traj_qm(self):
        """Get number of trajectories. Note that the order of trajectories in
        the dictionary is not the same as in Hamiltonian!
        The new_amp variable is non-zero only for the child and parent cloned
        TBFs. We assign amplitudes during the timestep after cloning happened,
        new_amp variable is zeroed out"""

        for ntraj, key in enumerate(self.traj):

            if self.traj_map[key] + 1 > len(self.qm_amplitudes):
                print "Adding trajectory ", key, "to the nuclear propagation"
                # Adding the quantum amplitude for a new trajectory
                self.qm_amplitudes = np.append(self.qm_amplitudes,
                                               self.traj[key].new_amp)
                self.traj[key].new_amp = 0.0

                for key2 in self.traj:
                    if np.abs(self.traj[key2].new_amp) > 1e-6\
                            and key2 != key:
                        # when new trajectory added we need
                        # to update the amplitude of parent
                        self.qm_amplitudes[self.traj_map[key2]]\
                            = self.traj[key2].new_amp
                        # zeroing out new_amp variable
                        self.traj[key2].new_amp = 0.0

        self.num_traj_qm = ntraj + 1

    def invert_S(self):
        """compute Sinv from S"""

        cond_num = np.linalg.cond(self.S)
        if cond_num > 500:
            print "BAD S matrix: condition number =", cond_num
        else:
            pass

        self.Sinv = np.linalg.inv(self.S)

    def build_Heff(self):
        """built Heff form H, Sinv, and Sdot"""

        c1i = (complex(0.0, 1.0))
        self.Heff = np.matmul(self.Sinv, (self.H - c1i * self.Sdot))

    def build_S_elec(self):
        """Build matrix of electronic overlaps"""

        ntraj = self.num_traj_qm
        self.S_elec = np.zeros((ntraj, ntraj), dtype=np.complex128)
        for keyi in self.traj:
            i = self.traj_map[keyi]
            if i < ntraj:
                for keyj in self.traj:
                    j = self.traj_map[keyj]
                    if j < ntraj:
                        if i == j:
                            self.S_elec[i, j] = 1.0
                        else:

                            wf_i_T = np.transpose(
                                np.conjugate(self.traj[keyi].td_wf_full_ts_qm))
                            wf_j = self.traj[keyj].td_wf_full_ts_qm
                            self.S_elec[i, j] = np.dot(wf_i_T, wf_j)

    def build_S(self):
        """Build the overlap matrix, S"""

        if self.quantum_time > 0.0:
            self.S_prev = self.S

        ntraj = self.num_traj_qm
        self.S = np.zeros((ntraj, ntraj), dtype=np.complex128)
        self.S_nuc = np.zeros((ntraj, ntraj), dtype=np.complex128)
        for keyi in self.traj:
            i = self.traj_map[keyi]
            if i < ntraj:
                for keyj in self.traj:
                    j = self.traj_map[keyj]
                    if j < ntraj:
                        self.S_nuc[i, j] = cg.overlap_nuc(
                            self.traj[keyi].positions_qm,
                            self.traj[keyj].positions_qm,
                            self.traj[keyi].momenta_qm,
                            self.traj[keyj].momenta_qm,
                            self.traj[keyi].widths,
                            self.traj[keyj].widths,
                            self.traj[keyi].numdims)

                        self.S[i, j] = self.S_nuc[i, j] * self.S_elec[i, j]

    def build_Sdot(self):
        """build the right-acting time derivative operator"""

        ntraj = self.num_traj_qm
        self.Sdot = np.zeros((ntraj, ntraj), dtype=np.complex128)
        self.S_dot_elec = np.zeros((ntraj, ntraj), dtype=np.complex128)
        self.S_dot_nuc = np.zeros((ntraj, ntraj), dtype=np.complex128)

        for keyi in self.traj:
            i = self.traj_map[keyi]
            if i < ntraj:
                for keyj in self.traj:
                    j = self.traj_map[keyj]
                    if j < ntraj:
                        self.S_dot_nuc[i, j] = cg.Sdot_nuc(
                            self.traj[keyi].positions_qm,
                            self.traj[keyj].positions_qm,
                            self.traj[keyi].momenta_qm,
                            self.traj[keyj].momenta_qm,
                            self.traj[keyi].widths,
                            self.traj[keyj].widths,
                            self.traj[keyj].av_force_qm,
                            self.traj[keyi].masses,
                            self.traj[keyi].numdims)

                        # Here we will call ES program to get Hamiltonian
                        H_elec = self.traj[keyj].construct_el_H(
                            self.traj[keyj].positions_qm)[0]
                        wf_j_dot = -1j\
                            * np.dot(H_elec, self.traj[keyj].td_wf_full_ts_qm)
                        wf_i_T = np.conjugate(
                            np.transpose(self.traj[keyi].td_wf_full_ts_qm))
                        self.S_dot_elec[i, j] = np.dot(wf_i_T, wf_j_dot)

                        self.Sdot[i, j] =\
                            np.dot(self.S_dot_elec[i, j], self.S_nuc[i, j])\
                            + np.dot(self.S_elec[i, j], self.S_dot_nuc[i, j])

    def build_H(self):
        """Building the Hamiltonian"""

        self.build_V()
        self.build_T()
        ntraj = self.num_traj_qm
        shift = self.qm_energy_shift * np.identity(ntraj)
        self.H = self.T + self.V + shift

    def build_V(self):
        """Build the potential energy matrix, V
        This routine assumes that S is already built"""

        ntraj = self.num_traj_qm
        self.V = np.zeros((ntraj, ntraj), dtype=np.complex128)
        for keyi in self.traj:
            i = self.traj_map[keyi]
            if i < ntraj:
                for keyj in self.traj:
                    j = self.traj_map[keyj]
                    if j < ntraj:
                        if i == j:
                            self.V[i, j] = self.traj[keyi].av_energy_qm
                        else:
                            nuc_overlap = self.S_nuc[i, j]

                            H_elec_i\
                                = self.traj[keyi].construct_el_H(
                                    self.traj[keyi].positions_qm)[0]

                            H_elec_j\
                                = self.traj[keyj].construct_el_H(
                                    self.traj[keyj].positions_qm)[0]

                            wf_i = self.traj[keyi].td_wf_full_ts_qm
                            wf_i_T = np.transpose(np.conjugate(wf_i))
                            wf_j = self.traj[keyj].td_wf_full_ts_qm
                            H_i = np.dot(wf_i_T, np.dot(H_elec_i, wf_j))
                            H_j = np.dot(wf_i_T, np.dot(H_elec_j, wf_j))
                            V_ij = 0.5 * (H_i + H_j)
                            self.V[i, j] = V_ij * nuc_overlap

    def build_T(self):
        "Building kinetic energy, needs electronic overlap S_elec"

        ntraj = self.num_traj_qm
        self.T = np.zeros((ntraj, ntraj), dtype=np.complex128)
        for keyi in self.traj:
            i = self.traj_map[keyi]
            if i < ntraj:
                for keyj in self.traj:
                    j = self.traj_map[keyj]
                    if j < ntraj:
                        self.T[i, j] = cg.kinetic_nuc(
                            self.traj[keyi].positions_qm,
                            self.traj[keyj].positions_qm,
                            self.traj[keyi].momenta_qm,
                            self.traj[keyj].momenta_qm,
                            self.traj[keyi].widths,
                            self.traj[keyj].widths,
                            self.traj[keyi].masses,
                            self.traj[keyi].numdims)\
                            * self.S_elec[i, j]

    def calc_approx_el_populations(self):
        """This calculates population on each electronic state taking into
        account all nuclear BFs. First we calculate nuclear population
        (Mulliken) out of two terms that cancel imaginary part.
        Then we multiply nuclear population by electronic population"""

        n_el_states = self.traj["00"].numstates
        norm = np.zeros(n_el_states)

        for n_el_state in range(n_el_states):
            ntraj = np.shape(self.S)[0]
            self.nuc_pop = np.zeros(ntraj)
            c_t = self.qm_amplitudes
            S_t = self.S
            for key1 in self.traj:
                nuc_pop = 0.0
                nuc_ind = self.traj_map[key1]
                for key2 in self.traj:
                    nuc_ind2 = self.traj_map[key2]

                    nuc_pop += np.real(
                        0.5 * (np.dot(np.conjugate(c_t[nuc_ind]),
                                      np.dot(S_t[nuc_ind, nuc_ind2],
                                             c_t[nuc_ind2]))
                               + np.dot(np.conjugate(c_t[nuc_ind2]),
                                        np.dot(S_t[nuc_ind2, nuc_ind],
                                               c_t[nuc_ind]))))
                self.nuc_pop[nuc_ind] = nuc_pop
                norm[n_el_state] += nuc_pop\
                    * self.traj[key1].populations[n_el_state]

            self.el_pop[n_el_state] = norm[n_el_state]

    def rescale_amplitudes(self, key, label, traj_1, traj_2,
                           ind_1, ind_2, new_dim, nuc_norm=1.0):
        """Here we need to rescale the coefficients of two cloned nuclear
        basis (1 and 2 in this notation) functions to conserve nuclear norm.
        The problem is that to do this we need overlaps with all other
        trajectories, which are not available yet.
        So we need to calculate all overlaps here, it is not a big deal
        because all trajectories and quantum amplitudes are propagated to
        the same time t"""

        amps = np.zeros((np.shape(self.S)[0] + 1),
                        dtype=np.complex128)
        S = np.zeros((np.shape(self.S)[0] + 1,
                      np.shape(self.S)[0] + 1),
                     dtype=np.complex128)

        # Here we rescale amplitudes right after cloning
        # These coefficients come from el norm conservation
        new_amp_1 = self.qm_amplitudes[ind_1] * traj_1.rescale_amp
        new_amp_2 = self.qm_amplitudes[ind_1] * traj_2.rescale_amp

        # check for overlap S12

        wf_1_T = np.transpose(np.conjugate(traj_1.td_wf_full_ts))
        wf_2 = traj_2.td_wf_full_ts
        S_elec_12 = np.dot(wf_1_T, wf_2)

        S_nuc_12 = cg.overlap_nuc(traj_1.positions, traj_2.positions,
                                  traj_1.momenta, traj_2.momenta,
                                  traj_1.widths, traj_2.widths,
                                  traj_1.numdims)

        S[ind_1, ind_2] = S_elec_12 * S_nuc_12
        S[ind_2, ind_1] = np.conjugate(S[ind_1, ind_2])

        pop_12n = 0.0
        pop_n = 0.0

        # this is population of the cloning BFs
        pop_12 = np.dot(np.conjugate(new_amp_1), new_amp_1)\
            + np.dot(np.conjugate(new_amp_2), new_amp_2)\
            + np.dot(np.conjugate(new_amp_1), new_amp_2)\
            * S[ind_1, ind_2]\
            + np.dot(np.conjugate(new_amp_2), new_amp_1)\
            * S[ind_2, ind_1]

        for ind in range(np.shape(self.S)[0] + 1):
            S[ind, ind] = 1.0

        if np.shape(self.S)[0] != 1:
            for key_n in self.traj:

                ind_n = self.traj_map[key_n]
                if key_n != key and key_n != label:
                    traj_n = self.traj[key_n]

                    amp_n = self.qm_amplitudes[ind_n]
                    amps[ind_n] = amp_n

                    wf_1 = traj_1.td_wf_full_ts
                    wf_1_T = np.transpose(np.conjugate(wf_1))

                    wf_n = traj_n.td_wf_full_ts
                    S_elec_1n = np.dot(wf_1_T, wf_n)
                    S_nuc_1n = cg.overlap_nuc(
                        traj_1.positions,
                        traj_n.positions,
                        traj_1.momenta,
                        traj_n.momenta_qm,
                        traj_1.widths,
                        traj_n.widths,
                        traj_1.numdims)

                    S[ind_1, ind_n] = S_elec_1n * S_nuc_1n
                    S[ind_n, ind_1] = np.conjugate(S[ind_1, ind_n])

                    wf_2 = traj_2.td_wf_full_ts
                    wf_2_T = np.transpose(np.conjugate(wf_2))

                    S_elec_2n = np.dot(wf_2_T, wf_n)
                    S_nuc_2n = cg.overlap_nuc(
                        traj_2.positions,
                        traj_n.positions,
                        traj_2.momenta,
                        traj_n.momenta_qm,
                        traj_2.widths,
                        traj_n.widths,
                        traj_2.numdims)

                    S[ind_2, ind_n] = S_elec_2n * S_nuc_2n
                    S[ind_n, ind_2] = np.conjugate(S[ind_2, ind_n])

                    # adding population from the overlap of cloning BFs
                    # with noncloning
                    pop_12n += np.dot(np.conjugate(new_amp_1), amps[ind_n])\
                        * S[ind_1, ind_n]\
                        + np.dot(np.conjugate(amps[ind_n]), new_amp_1)\
                        * S[ind_n, ind_1]\
                        + np.dot(np.conjugate(new_amp_2), amps[ind_n])\
                        * S[ind_2, ind_n]\
                        + np.dot(np.conjugate(amps[ind_n]), new_amp_2)\
                        * S[ind_n, ind_2]

                    # adding population from noncloning BFs,
                    # only diagonal contribution
                    pop_n += np.dot(np.conjugate(amp_n), amp_n)

                    for key_n2 in self.traj:

                        if key_n2 != key\
                            and key_n2 != label\
                                and key_n2 != key_n:

                            ind_n2 = self.traj_map[key_n2]
                            amp_n2 = self.qm_amplitudes[ind_n2]
                            amps[ind_n2] = amp_n2

                            S[ind_n, ind_n2] = self.S[ind_n, ind_n2]
                            S[ind_n2, ind_n] = self.S[ind_n2, ind_n]

                            # adding population of noncloning BFs,
                            # only off-diagonal contributions
                            pop_n += 1/4 * S[ind_n, ind_n2]\
                                * np.dot(np.conjugate(amps[ind_n]),
                                         amps[ind_n2]) + S[ind_n2, ind_n]\
                                * np.dot(np.conjugate(amps[ind_n2]),
                                         amps[ind_n])

        # this is the coefficient that ensures nuclear norm conservation
        rescale_coeff = (-pop_12n + np.sqrt(pop_12n**2
                         - 4 * (pop_n - nuc_norm) * pop_12)) / (2 * pop_12)
        alpha = np.dot(np.conjugate(rescale_coeff), rescale_coeff)
        if np.abs(alpha) < 1e-6:
            alpha = 1.0
        print "alpha =", alpha
        print "total_pop =", pop_n + rescale_coeff * pop_12n + pop_12 * alpha
        print "total pop before rescale =", pop_n + pop_12n + pop_12
        rescaled_amp_1 = new_amp_1 * rescale_coeff
        rescaled_amp_2 = new_amp_2 * rescale_coeff

        amps[ind_1] = traj_1.new_amp
        amps[ind_2] = traj_2.new_amp
        norm = np.dot(np.conjugate(np.transpose(amps)), np.dot(S, amps))
        print "Norm = ", norm

        # Number overlap elements that are larger than threshold
        # If any off diagonal element is larger, we abort cloning
        s = (abs(S) > self.olapmax).sum()
#         print "SUM =", s
#         print "S =", np.abs(S[0,:])
        abort_clone = False
        if s > new_dim:
            abort_clone = True

        return rescaled_amp_1, rescaled_amp_2, abort_clone

    def build_Heff_half_timestep(self):
        """build Heff for the either half of the time step in the diabatic rep
        Since we don't need any information at half time step there is no
        difference between first and second half"""

        self.get_qm_data_from_h5()
        self.build_S_elec()
        self.build_S()
        self.invert_S()
        self.build_Sdot()
        self.build_H()
        self.build_Heff()

    def clone_to_a_state(self):
        """Cloning routine. Trajectories that are cloning will be established
        from the cloning probabilities variable clone_e_gap.
        When a new basis function is added the labeling is done
        in a following way: a trajectory labeled 00b1b5 means
        that the initial trajectory "00" cloned a trajectory "1"
        (its second child) which then spawned another (it's 6th child)"""

        clonetraj = dict()
        for key in self.traj:
            # sorting states according to decreasing gap * el population
            if self.traj[key].full_H:
                istates = (-self.traj[key].clone_e_gap[:]
                           * self.traj[key].populations[:]).argsort()
            else:
                istates = (-self.traj[key].clone_e_gap[:]
                           * self.traj[key].approx_pop[:]).argsort()
            # nuclear population of this trajectory
            nuc_pop = self.nuc_pop[self.traj_map[key]]

            if nuc_pop < self.nuc_pop_thresh:
#                 print "Trajectory " + key + " can't clone because there is only "\
#                 + str(round(nuc_pop*100, 2)) + "% of nuclear population on it"
                continue  # if nuclear popless than threshold -> do not clone

            for istate in istates:
                """Do we clone to istate?
                If the following conditions are satisfied then we clone
                to that state and we're done. If done we go to another
                state with lower probability"""

                if self.traj[key].full_H:
                    # If we use real eigenstates from H diagonalization
                    pop_to_check = self.traj[key].populations
                else:
                    # If we use approximate eigenstates
                    pop_to_check = self.traj[key].approx_pop
                clone_now = True
                # Energy of a specific state is larger than threshold
                clone_now *= self.traj[key].clone_e_gap[istate]\
                    > self.e_gap_thresh
                # Population is larger than threshold
                clone_now *= pop_to_check[istate] > self.pop_threshold
                # This is arbitrary but we clone to the state which has low pop
                clone_now *= pop_to_check[istate] < 1.0 - self.pop_threshold
                # This makes sure we clone only if energy gap is increasing
                clone_now *= self.traj[key].clone_e_gap[istate]\
                    >= self.traj[key].clone_e_gap_prev[istate]

                if clone_now:

                    print "Trajectory " + key + " with nuclear population of "\
                        + str(round(nuc_pop*100, 2)) + "% cloning to "\
                        + str(istate) + " state at time "\
                        + str(self.traj[key].time)
                    print "with p = " + str(self.traj[key].clone_e_gap[istate])
                    print "Electronic populations on trajectory " + key + ":"
                    print self.traj[key].approx_pop

                    label = str(self.traj[key].label) + "b" +\
                        str(self.traj[key].numchildren)
                    # create and initiate new trajectory structure
                    newtraj = traj(self.traj[key].numdims,
                                   self.traj[key].numstates,
                                   self.traj[key].krylov_sub_n)

                    # making a copy of a parent BF in order not to overwrite
                    # the original in case cloning fails due to large overlap
                    parent_copy = copy.deepcopy(self.traj[key])

                    # okay, now we finally decide whether to clone or not
                    if parent_copy.full_H:
                        # cloning based on real eigenstates
                        clone_ok = newtraj.init_clone_traj_to_a_state(
                            parent_copy, istate, label)
                    else:
                        # cloning based on approximate eigenstates
                        clone_ok = newtraj.init_clone_traj_approx(
                            parent_copy, istate, label)

                    if clone_ok:

                        # the dimensionality increases when we add new function
                        # but the matrices are not updated yet
                        new_dim = np.shape(self.S)[0] + 1
                        ind_1 = self.traj_map[key]  # index of parent
                        ind_2 = np.shape(self.S)[0]  # index of child
                        traj_1 = parent_copy
                        traj_2 = newtraj
                        # Calculating rescaled amplitudes and figure out
                        # if there is large overlap with other trajectories
                        rescaled_amp_1, rescaled_amp_2, abort_clone\
                            = self.rescale_amplitudes(
                                key, label, traj_1, traj_2, ind_1, ind_2,
                                new_dim, nuc_norm=1.0)

                        traj_1.new_amp = rescaled_amp_1
                        traj_2.new_amp = rescaled_amp_2

                        if abort_clone:
                            print "Aborting cloning due to large overlap with",
                            " existing trajectory"
                            self.clone_again = False
                            continue

                        else:
                            # if overlap is less than threshold
                            # we can update the actual parent
                            self.traj[key] = traj_1
                            print "Overlap OK, creating new trajectory ", label
                            clonetraj[label] = traj_2
                            self.traj[key].numchildren += 1
                            self.add_traj(clonetraj[label])
                            # update matrices here in case
                            # other trajectories clone
                            self.compute_num_traj_qm()
                            self.build_Heff_half_timestep()
                            self.calc_approx_el_populations()
                            self.clone_again = True
                            return

                    else:
                        self.clone_again = False
                        continue

    def restart_from_file(self, json_file, h5_file):
        """restarts from the current json file and copies
        the simulation data into working.hdf5"""

        self.read_from_file(json_file)
        shutil.copy2(h5_file, "working.hdf5")

    def restart_output(self):
        """output json restart file
        The json file is meant to represent the *current* state of the
        simulation.  There is a separate hdf5 file that stores the history of
        the simulation"""

#         print "Creating new sim.json"
#         we keep copies of the last 3 json files just to be safe
        extensions = [3, 2, 1, 0]
        for i in extensions:
            if i == 0:
                ext = ""
            else:
                ext = str(i) + "."
            filename = "sim." + ext + "json"
            if os.path.isfile(filename):
                if i == extensions[0]:
                    os.remove(filename)
                else:
                    ext = str(i+1) + "."
                    filename2 = "sim." + ext + "json"
                    if i == extensions[-1]:
                        shutil.copy2(filename, filename2)
                    else:
                        shutil.move(filename, filename2)

        # now we write the current json file
        self.write_to_file("sim.json")
        extensions = [3, 2, 1, 0]
        for i in extensions:
            if i == 0:
                ext = ""
            else:
                ext = str(i) + "."
            filename = "sim." + ext + "hdf5"
            if os.path.isfile(filename):
                if i == extensions[0]:
                    os.remove(filename)
                else:
                    ext = str(i + 1) + "."
                    filename2 = "sim." + ext + "hdf5"
                    if i == extensions[-1]:
                        shutil.copy2(filename, filename2)
                    else:
                        shutil.move(filename, filename2)
        shutil.copy2("working.hdf5", "sim.hdf5")
        print "hdf5 and json output are synchronized"

    def h5_output(self):
        """"Writes output  to h5 file"""

        self.init_h5_datasets()
        filename = "working.hdf5"
        h5f = h5py.File(filename, "a")
        groupname = "sim"
        if groupname not in h5f.keys():
            # creating sim group in hdf5 output file
            self.create_h5_sim(h5f, groupname)
            grp = h5f.get(groupname)
            self.create_new_h5_map(grp)
        else:
            grp = h5f.get(groupname)
        znewmap = False
        for key in self.h5_datasets:
            n = self.h5_datasets[key]
            dset = grp.get(key)
            length = dset.len()
            if length > 0:
                lwidth = dset.size / length
                if n > lwidth:
                    dset.resize(n, axis=1)
                    if not znewmap:
                        self.create_new_h5_map(grp)
                        znewmap = True
            dset.resize(length + 1, axis=0)
            ipos = length
#             getcom = "self.get_" + key + "()"
            getcom = "self." + key
#             print getcom

            tmp = eval(getcom)
            if type(tmp).__module__ == np.__name__:
                tmp = np.ndarray.flatten(tmp)
                dset[ipos, 0:n] = tmp[0:n]
            else:
                dset[ipos, 0] = tmp
        h5f.flush()
        h5f.close()

    def create_new_h5_map(self, grp):
        """Creates a map from trajectory labels to the dimensions
        of the nuclear Hamiltonian. The order of trajectories
        in the dictionary of labels is not the same as it is
        in Hamiltonian"""

        ntraj = self.num_traj_qm
        labels = np.empty(ntraj, dtype="S512")
        for key in self.traj_map:
            if self.traj_map[key] < ntraj:
                labels[self.traj_map[key]] = key
        grp.attrs["labels"] = labels
        grp.attrs["olapmax"] = self.olapmax
        grp.attrs["e_gap_thresh"] = self.e_gap_thresh
        grp.attrs["pop_threshold"] = self.pop_threshold
        grp.attrs['nuc_pop_thresh'] = self.nuc_pop_thresh

    def create_h5_sim(self, h5f, groupname):
        """Creates h5 datasets"""

        trajgrp = h5f.create_group(groupname)
        for key in self.h5_datasets:
            n = self.h5_datasets[key]
            trajgrp.create_dataset(key, (0, n), maxshape=(None, None),
                                   dtype=self.h5_types[key])

    def init_h5_datasets(self):
        """Initialization of the h5 datasets within the simulation object"""

        ntraj = self.num_traj_qm
        ntraj2 = ntraj * ntraj

        self.h5_datasets = dict()
        self.h5_datasets["quantum_time"] = 1
        self.h5_datasets["qm_amplitudes"] = ntraj
        self.h5_datasets["Heff"] = ntraj2
        self.h5_datasets["H"] = ntraj2
        self.h5_datasets["S"] = ntraj2
        self.h5_datasets["Sdot"] = ntraj2
        self.h5_datasets["Sinv"] = ntraj2
        self.h5_datasets["num_traj_qm"] = 1
        self.h5_datasets["el_pop"] = self.num_el_states

        self.h5_types = dict()
        self.h5_types["quantum_time"] = "float64"
        self.h5_types["qm_amplitudes"] = "complex128"
        self.h5_types["Heff"] = "complex128"
        self.h5_types["H"] = "complex128"
        self.h5_types["S"] = "complex128"
        self.h5_types["Sdot"] = "complex128"
        self.h5_types["Sinv"] = "complex128"
        self.h5_types["num_traj_qm"] = "int32"
        self.h5_types["el_pop"] = "float64"

    def get_qm_data_from_h5(self):
        """get the necessary geometries and energies from hdf5"""

        qm_time = self.quantum_time
        ntraj = self.num_traj_qm
        for key in self.traj:
            if self.traj_map[key] < ntraj:
                self.traj[key].get_all_qm_data_at_time_from_h5(qm_time)

    def get_numtasks(self):
        """get the number of tasks in the queue"""

        return len(self.queue) - 1

    def pop_task(self):
        """pop the task from the top of the queue"""

        return self.queue.pop(0)

    def update_queue(self):
        """build a list of all tasks that need to be completed"""
        while self.queue[0] != "END":
            self.queue.pop(0)
        tasktimes = [1e10]

        # forward propagation tasks
        for key in self.traj:
            if (self.traj[key].maxtime + 1.0e-6) > self.traj[key].time:
                task_tmp = "self.traj[\"" + key + "\"].propagate_step()"
                tasktime_tmp = self.traj[key].time
                self.insert_task(task_tmp, tasktime_tmp, tasktimes)

        print (len(self.queue)-1), "task(s) in queue:"
        for i in range(len(self.queue)-1):
            print self.queue[i] + ", time = " + str(tasktimes[i])
        print ""

    def insert_task(self, task, tt, tasktimes):
        """Add a task to the queue"""

        for i in range(len(tasktimes)):
            if tt < tasktimes[i]:
                self.queue.insert(i, task)
                tasktimes.insert(i, tt)
                return

    def from_dict(self, **tempdict):
        """Converts dict to simulation data structure
        This is used for restart"""

        for key in tempdict:
            if isinstance(tempdict[key], types.UnicodeType):
                tempdict[key] = str(tempdict[key])
            if isinstance(tempdict[key], types.ListType):
                if isinstance((tempdict[key])[0], types.FloatType):
                    # convert 1d float lists to np arrays
                    tempdict[key] = np.asarray(tempdict[key])
                if isinstance((tempdict[key])[0], types.StringTypes):
                    if (tempdict[key])[0][0] == "^":
                        for i in range(len(tempdict[key])):
                            tempdict[key][i] = eval(tempdict[key][i][1:])
                        tempdict[key] = np.asarray(tempdict[key],
                                                   dtype=np.complex128)
                else:
                    if isinstance((tempdict[key])[0], types.ListType):
                        if isinstance((tempdict[key])[0][0], types.FloatType):
                            # convert 2d float lists to np arrays
                            tempdict[key] = np.asarray(tempdict[key])
                        if isinstance((tempdict[key])[0][0],
                                      types.StringTypes):
                            if (tempdict[key])[0][0][0] == "^":
                                for i in range(len(tempdict[key])):
                                    for j in range(len(tempdict[key][i])):
                                        tempdict[key][i][j]\
                                            = eval(tempdict[key][i][j][1:])
                                tempdict[key] = np.asarray(tempdict[key],
                                                           dtype=np.complex128)
            if isinstance(tempdict[key], types.DictType):
                if 'fmsobjlabel' in (tempdict[key]).keys():
                    fmsobjlabel = (tempdict[key]).pop('fmsobjlabel')
                    obj = eval(fmsobjlabel[8:])()
                    obj.from_dict(**(tempdict[key]))
                    tempdict[key] = obj
                else:
                    for key2 in tempdict[key]:
                        if isinstance((tempdict[key])[key2], types.DictType):
                            if key == 'traj':
                                # This is a hack that fixes the previous hack
                                # initially trajectory's init didn't have
                                # numstates and numdims which caused certain
                                # issues so I'm adding the variables for traj
                                # initialization to make restart work
                                numdims = tempdict[key][key2]['numdims']
                                numstates = tempdict[key][key2]['numstates']
                                krylov_sub_n\
                                    = tempdict[key][key2]['krylov_sub_n']
                                fmsobjlabel\
                                    = ((tempdict[key])[key2]).pop(
                                        'fmsobjlabel')
                                obj = eval(fmsobjlabel[8:])(numdims, numstates,
                                                            krylov_sub_n)
                                obj.from_dict(**((tempdict[key])[key2]))
                                (tempdict[key])[key2] = obj
                            else:
                                fmsobjlabel = (
                                    (tempdict[key])[key2]).pop('fmsobjlabel')
                                obj = eval(fmsobjlabel[8:])()
                                obj.from_dict(**((tempdict[key])[key2]))
                                (tempdict[key])[key2] = obj
        self.__dict__.update(tempdict)
