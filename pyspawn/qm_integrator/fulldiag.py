import numpy as np
import numpy.linalg as la
import math


def qm_propagate_step(self, zoutput_first_step=False):
    """Exponential integrator (***Needs reference***)"""

    c1i = (complex(0.0, 1.0))
    self.compute_num_traj_qm()
    qm_t = self.quantum_time
    dt = self.timestep
    qm_tpdt = qm_t + dt 
    ntraj = self.num_traj_qm

    amps_t = self.qm_amplitudes
    print "Building effective Hamiltonian for the first half step"
    self.build_Heff_half_timestep()
    self.calc_approx_el_populations()
    norm = np.dot(np.conjugate(np.transpose(amps_t)), np.dot(self.S, amps_t))
    #print "Norm first half =", norm    
    
    # output the first step before propagating
    if zoutput_first_step:
        self.h5_output()

    iHdt = (-0.5 * dt * c1i) * self.Heff
    W,R = la.eig(iHdt)
    X = np.exp( W )
    amps = amps_t
    tmp1 = la.solve(R, amps)
    tmp2 = X * tmp1 # element-wise multiplication
    amps = np.matmul(R, tmp2)

    self.quantum_time = qm_tpdt
    print "Building effective Hamiltonian for the second half step"
    self.build_Heff_half_timestep()
    print "Effective Hamiltonian built"    
 
    iHdt = (-0.5 * dt * c1i) * self.Heff
    W,R = la.eig(iHdt)
    X = np.exp(W)
    tmp1 = la.solve(R, amps)
    tmp2 = X * tmp1 # element-wise multiplication
    amps = np.matmul(R, tmp2)

    self.qm_amplitudes = amps 
    norm = np.dot(np.conjugate(np.transpose(amps)), np.dot(self.S, amps))
    #print "Norm second half =", norm
    if abs(norm-1.0) > 0.01:
        print "Warning: nuclear norm deviated from 1: norm =", norm 
    print "Done with quantum propagation"            
