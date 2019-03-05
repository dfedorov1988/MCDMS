# this script starts a new FMS calculation on a model cone potential
import numpy as np
import pyspawn        
import pyspawn.general
import os
p_thresh_list = [0.06, 0.05, 0.04]
pop_thresh_list = [0.05, 0.03, 0.01]
krylov_sub_n_list = [2, 3, 4, 5, 6, 7, 8, 9]

for krylov_sub_n in krylov_sub_n_list:
    for pop_thresh in pop_thresh_list:
        for p_thresh in p_thresh_list:
            # Nuclear population threshold, if population on a specific basis function is smaller -> no cloning
            nuc_pop_thresh = 0.05
            # Minimum population of electronic state for cloning to occur to it
#             pop_thresh = 0.03
            # Minimum energy gap for cloning to occur
#             p_thresh = 0.06
            # Maximum overlap threshold, if higher -> no cloning
            olapmax = 0.5
            # Use real eigenstates or approximate (Krylov subspace)
            full_H = False
            # Size of Krylov subspace
#             krylov_sub_n = 5
            # Velocity Verlet classical propagator
            clas_prop = "vv"
            # fulldiag exponential quantum propagator
            qm_prop = "fulldiag"
            # diabatic ehrenfest Hamiltonian
            qm_ham = "ehrenfest"
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
            numstates = 9
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
                # inition momenta
                "momenta": np.asarray([10.0]),
                "full_H": full_H,
                "numstates": numstates,
                "n_el_steps": n_el_steps,    
                }
            
            sim_params = {
                "quantum_time": traj_params["time"],
                "timestep": traj_params["timestep"],
                "max_quantum_time": traj_params["maxtime"],
                "qm_amplitudes": np.ones(1, dtype=np.complex128),
                # energy shift used in quantum propagation
                "qm_energy_shift": 0.0,
                "p_threshold": p_thresh,
                "olapmax": olapmax,
                "pop_threshold": pop_thresh,
                "nuc_pop_thresh": nuc_pop_thresh,
                # type of cloning procedure:
                # "toastate" : cloning on to a state energy of which is different from average
                # "pairwise" : considering each pair, transferring population between them
                "cloning_type": "toastate",
            }
            
            # import routines needed for propagation
            exec("pyspawn.import_methods.into_simulation(pyspawn.qm_integrator." + qm_prop + ")")
            exec("pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian." + qm_ham + ")")
            exec("pyspawn.import_methods.into_traj(pyspawn.potential." + potential + ")")
            exec("pyspawn.import_methods.into_traj(pyspawn.classical_integrator." + clas_prop + ")")
            
            # check for the existence of files from a past run
            pyspawn.general.check_files()    
            
            # set up first trajectory
            if full_H: krylov_sub_n = numstates
            traj1 = pyspawn.traj(numdims, numstates, krylov_sub_n)
            traj1.set_parameters(traj_params)
            
            # set up simulation 
            sim = pyspawn.simulation()
            sim.add_traj(traj1)
            sim.set_parameters(sim_params)
            # begin propagation
            sim.propagate()
            os.system('python analysis.py >> output')
            
