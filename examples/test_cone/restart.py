# this script restarts the simulation using data from sim.json (which 
# contains the entire current state of the simulation) and sim.hdf5 (which 
# contains a selected history of the simulation
import numpy as np
import pyspawn         

tfinal = 110.0

sim = pyspawn.Simulation(17)

sim.restart_from_file("sim.json", "sim.hdf5")

sim.set_maxtime_all(tfinal)

sim.propagate()







