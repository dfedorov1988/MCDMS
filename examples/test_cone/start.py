# this script starts a new FMS calculation on a model cone potential
import numpy as np
import pyspawn
import pyspawn.general
import os

# Nuclear population threshold, if population on a specific basis function
# is smaller -> no cloning
nuc_pop_thresh = 0.05
# Minimum population of electronic state for cloning to occur to it
pop_thresh = 0.03
# Minimum energy gap for cloning to occur
e_gap_thresh = 0.05
# Maximum overlap threshold, if higher -> no cloning
olapmax = 0.5
# Use real eigenstates or approximate (Krylov subspace)
full_H = False
# Size of Krylov subspace
krylov_sub_n = 5
# Velocity Verlet classical propagator
clas_prop = "vv"
# fulldiag exponential quantum propagator
qm_prop = "fulldiag"
# use TeraChem CASSCF or CASCI to compute potentials
potential = "linear_slope"
# initial time
t0 = 0.0
# time step
ts = 0.1
# final simulation time
tfinal = 80.0
# number of dimensions                                                                                           
numdims = 1
# number of electronic states                                                                                                           
numstates = 5
# number of electronic timesteps within one nuclear
n_el_steps = 100

# trajectory parameters
traj_params = {
    "time": t0,
    "timestep": ts,
    "maxtime": tfinal,
    # Gaussian widths
    "widths": np.asarray([6.0]),
    # nuclear masses (in a.u)    
    "masses": np.asarray([1822.0]),
    # initial positions
    "positions": np.asarray([-0.2]),
    # initial momenta
    "momenta": np.asarray([10.0]),
    "full_H": full_H,
    "numstates": numstates,
    "n_el_steps": n_el_steps,
#     "potential": "linear_slope"
    }

sim_params = {
    "quantum_time": traj_params["time"],
    "timestep": traj_params["timestep"],
    "max_quantum_time": traj_params["maxtime"],
    "qm_amplitudes": np.ones(1, dtype=np.complex128),
    # energy shift used in quantum propagation
    "qm_energy_shift": 0.0,
    "e_gap_thresh": e_gap_thresh,
    "olapmax": olapmax,
    "pop_threshold": pop_thresh,
    "nuc_pop_thresh": nuc_pop_thresh,
    "num_el_states": numstates,
    # type of cloning procedure:
    # "toastate" : cloning on to a state energy of which is different from average
    # "pairwise" : considering each pair, transferring population between them
    "cloning_type": "toastate",
}

# import routines needed for propagation
# exec("pyspawn.import_methods.into_simulation(pyspawn.qm_integrator." + qm_prop + ")")
# exec("pyspawn.import_methods.into_traj(pyspawn.potential." + potential + ")")
# exec("pyspawn.import_methods.into_traj(pyspawn.classical_integrator." + clas_prop + ")")

# check for the existence of files from a past run
pyspawn.general.check_files()    

# set up first trajectory
if full_H:
    krylov_sub_n = numstates
traj1 = pyspawn.traj(numdims, numstates, krylov_sub_n)
traj1.set_parameters(traj_params)

# set up simulation 
sim = pyspawn.Simulation(numstates)
sim.add_traj(traj1)
sim.set_parameters(sim_params)
# begin propagation
sim.propagate()
os.system('python analysis.py >> output')
