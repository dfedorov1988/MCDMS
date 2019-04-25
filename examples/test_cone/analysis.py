"""this analysis script processes the sim.hdf5 file into various human-readable
formats.  This script can be run while the simulation is in progress"""

import pyspawn
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import glob


def plot_nuclear_populations(ntraj, linestyles, labels, markers):
    """Plots nuclear basis functions' contributions to the total nuclear wf"""

    g3 = plt.figure("Nuclear Populations")
    N = an.datasets["nuclear_bf_populations"]
    qm_time = an.datasets["quantum_times"]
    for n in range(ntraj):
        plt.plot(qm_time, N[:, n + 1], linestyle=linestyles[n],
                 marker=markers[n], markersize=3, markevery=15,
                 label=labels[n])
    plt.xlabel('Time, au', fontsize=medium_size)
    plt.ylabel('Nuclear Population', fontsize=medium_size)
    plt.legend(fontsize=medium_size)
    plt.tick_params(axis='both', which='major', labelsize=small_size)
    plt.title('Nuclear Population', fontsize=large_size)
    plt.tight_layout()
    g3.savefig("Nuc_pop.png", dpi=300)


def plot_el_population(time, pop, keys, numstates, colors,
                       linestyles, markers):
    '''Plots electronic state populations for all trajectories,
    messy when a lot of states/trajectories'''

    g1 = plt.figure("Electronic Populations")
    for n_key, key in enumerate(keys):
        for n_state in range(numstates):
            plt.plot(time[key], pop[key][:, n_state], color=colors[n_state],
                     label=key + ": " + str((n_state + 1)) + " state",
                     linestyle=linestyles[n_key], marker=markers[n_key],
                     markersize=2, markevery=20)

    plt.xlabel('Time, au', fontsize=medium_size)
    plt.ylabel('Population', fontsize=medium_size)
    plt.legend(fontsize=medium_size)
    plt.title('Electronic Populations', fontsize=large_size)
    plt.tick_params(axis='both', which='major', labelsize=small_size)
    g1.savefig("Elec_pop.png", dpi=300)


def plot_energies(time, poten, keys, numstates, colors, linestyles, markers):
    """Plots energy of each eigenstate (from diagonalization of H)
    and Ehrenfest energy for each trajectory"""

    g2 = plt.figure("Energies")
#     Plotting eigenstate energies only for first trajectory for clarity
    key = '00'
    n_key = 0
    for n_state in range(numstates):
        label = "S" + str(n_state)
        plt.plot(time[key][::10], poten[key][::10, n_state],
                 color=colors[n_state],
                 label=label,
                 linestyle=linestyles[n_key], marker=markers[n_key],
                 alpha=0.5, markersize=3, markevery=2)

    for n_key, key in enumerate(keys):
        plt.plot(time[key], aven[key], color='black',
                 label='TBF' + str(n_key),
                 linestyle=linestyles[n_key], marker=markers[n_key],
                 markersize=3, markevery=20)

    plt.xlabel('Time, au', fontsize=medium_size)
    plt.ylabel('Energy, au', fontsize=medium_size)
    plt.legend(fontsize=small_size-2)
    plt.title('Eigenstate Energies', fontsize=large_size)
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=small_size)
    plt.text(time['00'][-1] - 20, poten['00'][-1, 0] + 0.03,
             str(round(pop_band[0]*100)) + "%", fontsize=medium_size)
#     plt.text(time['00'][-1] - 10, poten['00'][-1, 1] - 0.03,
#              str(round(pop_band[1] * 100)) + "%", fontsize=medium_size)
#     plt.text(time['00'][-1] - 10, poten['00'][-1, 5] - 0.03,
#              str(round(pop_band[2] * 100)) + "%", fontsize=medium_size)
    g2.savefig("Energies.png", dpi=300)


def plot_approx_energies(keys, numstates, colors, linestyles, markers):
    """Plots approximate eigenstates energies along with the Ehrenfest
    energies,\n useful to check what's going on
    with approximate eigenstates"""

    g2 = plt.figure(" Approximate Eigenstates' Energies")
    n_key = 0
    key = '00'
    approx_e = trajfile['traj_' + key]['approx_energies']
    for n_state in range(numstates):
        if n_key == 0:
            cur_label = 'S' + str(n_state)
        else:
            cur_label = None
        plt.plot(
            time[key][::10], approx_e[::10, n_state],
            color=colors[n_state],
            label=cur_label,
            linestyle=linestyles[n_key],
            marker=markers[n_key], alpha=0.5, markersize=3,
            markevery=2)

    for n_key, key in enumerate(keys):
        plt.plot(time[key], aven[key], color='black',
                 label='TBF' + str(n_key), linestyle=linestyles[n_key],
                 marker=markers[n_key], markersize=3, markevery=20)

    plt.xlabel('Time, au', fontsize=medium_size)
    plt.ylabel('Approximate Energy, au', fontsize=medium_size)
    plt.text(time['00'][-1] - 20, poten['00'][-1, 0] + 0.03,
             str(round(pop_band[0] * 100)) + "%", fontsize=medium_size)
#    plt.text(time['00'][-1] - 10, poten['00'][-1, 1] - 0.03,
#             str(round(pop_band[1]*100)) + "%", fontsize=medium_size)
#    plt.text(time['00'][-1] - 10, poten['00'][-1, 5] - 0.03,
#             str(round(pop_band[2]*100)) + "%", fontsize=medium_size)
    plt.legend(fontsize=medium_size)
    plt.title('Energies of Approximate Eigenstates', fontsize=large_size)
    plt.tick_params(axis='both', which='major', labelsize=small_size)
    plt.tight_layout()
    g2.savefig("Approx_energies.png", dpi=300)


def plot_real_wf(keys, numstates, krylov_sub_n):

    g10 = plt.figure("Real Part of Electronic wf")
    key = keys[0]
    time_shape = time[key].shape[0]
    wf_store = trajfile['traj_' + key]['wf_store']
    wf_store_2d = np.reshape(wf_store, (time_shape, numstates, krylov_sub_n))
    for n_state in range(krylov_sub_n):
        plt.plot(time[key], wf_store_2d[:, 0, n_state],
                 label=key + ' ' + str(n_state) + " eigenstate")
    plt.legend(fontsize=medium_size)
    plt.tick_params(axis='both', which='major', labelsize=small_size)
    plt.title('Real Part of Wave Function', fontsize=large_size)
#     plt.tight_layout()
    g10.savefig("Real_wf.png", dpi=300)


def plot_approx_wf(keys, krylov_sub_n):

    g10 = plt.figure("Real Part of Electronic wf")
    key = keys[0]
    wf = trajfile['traj_' + key]['approx_wf_full_ts']
    for n_state in range(krylov_sub_n):
        plt.plot(time[key], np.imag(wf[:, n_state]),
                 label=key + ' ' + str(n_state) + " eigenstate")
    plt.legend(fontsize=medium_size)
    plt.tick_params(axis='both', which='major', labelsize=small_size)
    plt.title('Approximate Wave Function', fontsize=large_size)
#     plt.tight_layout()
    g10.savefig("Approx_wf.png", dpi=300)


def plot_approx_el_population(time, keys, numstates, colors):
    """Prints approximate electronic populations for trajectories in keys,
    saves all of them into separate files"""

    for key in keys:
        fig = plt.figure("Approximate Electronic Populations " + key)
        approx_pop = trajfile['traj_' + key]['approx_pop']
        for n_state in range(numstates):
            plt.plot(time[key], approx_pop[:, n_state], color=colors[n_state],
                     label=key + ": " + str((n_state+1)) + " state",
                     linestyle='-')
        plt.xlabel('Time, au', fontsize=medium_size)
        plt.ylabel('Population', fontsize=medium_size)
        plt.tick_params(axis='both', which='major', labelsize=small_size)
        plt.legend(fontsize=medium_size)
        plt.title('Approximate States Electronic Population',
                  fontsize=large_size)
        plt.tight_layout()
        fig.savefig("Approx_elec_pop_" + key + ".png", dpi=300)


def plot_total_energies(time, toten, keys):
    """Plots total classical energies for each trajectory,
    useful to look at energy conservation"""

    g4 = plt.figure("Total Energies")
    min_E = min(toten["00"])
    max_E = max(toten["00"])
    for key in keys:
        plt.plot(time[key], toten[key],
                 label=key)
        if min(toten[key]) < min_E:
            min_E = min(toten[key])
        if max(toten[key]) > max_E:
            max_E = max(toten[key])

    plt.xlabel('Time, au', fontsize=medium_size)
    plt.ylabel('Total Energy, au', fontsize=medium_size)
    plt.ylim([min_E - 0.05 * (max_E - min_E), max_E + 0.05 * (max_E - min_E)])
    plt.legend(fontsize=medium_size)
    plt.tick_params(axis='both', which='major', labelsize=small_size)
    plt.title('Total Energies', fontsize=large_size)
    plt.tight_layout()
    g4.savefig("Total_Energies.png", dpi=300)


def plot_total_pop():
    """ This plots the total electronic population on each
    electronic state (over all basis functions)"""

    g5 = plt.figure("Total Electronic Populations")
    for n_state in range(nstates):
        plt.plot(time["00"][:], el_pop[:, n_state], color=colors[n_state],
                 label='S' + str(n_state))
    plt.xlabel('Time, au', fontsize=medium_size)
    plt.ylabel('Population', fontsize=medium_size)
    plt.text(time['00'][-1] - 13, el_pop[-1, 0] - 0.1,
             str(round(el_pop[-1, 0] * 100)) + "%", fontsize=medium_size)
    plt.title('Total Electronic Population', fontsize=large_size)
    plt.tick_params(axis='both', which='major', labelsize=small_size)
    plt.legend(fontsize=medium_size)
    plt.tight_layout()
    g5.savefig("Total_El_pop.png", dpi=300)


an = pyspawn.fafile("sim.hdf5")
work = pyspawn.fafile("working.hdf5")
# create N.dat and store the data in times and N
an.fill_nuclear_bf_populations()
an.fill_trajectory_populations()
an.fill_labels()
# write files with energy data for each trajectory
an.fill_trajectory_energies()
# list all datasets
# an.list_datasets()

large_size = 16
medium_size = 14
small_size = 12

ntraj = len(an.datasets["labels"])
colors = ("r", "g", "b", "m", "y", "tab:purple", 'xkcd:sky blue',
          "xkcd:teal blue", 'xkcd:puce', 'k')
linestyles = ("-", "--", "-.", ":", "-", "-", "-", "-", "-", "-",
              "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-")
markers = ("None", "None", "None", "None", "d", "o", "v", "^", "s", "p", "d",
           "o", "v", "^", "s", "p", "d", "o", "v", "^", "s", "p", "d", "o",
           "v", "^", "s", "p")
labels = an.datasets["labels"]
el_pop = an.datasets["el_pop"]

h5filename = "sim.hdf5"
h5file = h5py.File(h5filename, "r")
trajfilename = "working.hdf5"
trajfile = h5py.File(trajfilename, "r")

# Reading attributes from trajectories, simulation to use for creating
# directories when multiple simulations are run
full_H = trajfile["traj_00"].attrs["full_H"]
nstates = trajfile["traj_00"].attrs["numstates"]
krylov_sub_n = trajfile["traj_00"].attrs["krylov_sub_n"]
e_gap_thresh = h5file["sim"].attrs['e_gap_thresh']
pop_threshold = h5file["sim"].attrs['pop_threshold']
olapmax = h5file["sim"].attrs["olapmax"]
nuc_pop_thresh = h5file["sim"].attrs['nuc_pop_thresh']

# Initiating dictionaries for the arrays we need for analysis
poten = {}
pop = {}
toten = {}
aven = {}
kinen = {}
time = {}

for traj in an.datasets["labels"]:

    poten[traj] = an.datasets[traj + "_poten"]
    pop[traj] = an.datasets[traj + "_pop"]
    toten[traj] = an.datasets[traj + "_toten"]
    aven[traj] = an.datasets[traj + "_aven"]
    kinen[traj] = an.datasets[traj + "_kinen"]
    time[traj] = an.datasets[traj + "_time"]

# Summing populations over parallel states/bands (hardcoded for 3 bands
# in 9 state system

pop_band = np.zeros(3)
pop_band[0] = el_pop[-1, 0]
pop_band[1] = sum(el_pop[-1, 1:5])
pop_band[2] = sum(el_pop[-1, 5:9])
np.savetxt('pop_band.dat', pop_band)

# Plotting
if time['00'].shape[0] == el_pop[:, 0].shape[0]:
    plot_total_pop()
plot_el_population(time, pop, ['00'], nstates, colors, linestyles, markers)
plot_energies(time, poten, labels, nstates, colors, linestyles, markers)
plot_approx_energies(an.datasets["labels"], krylov_sub_n, colors,
                     linestyles, markers)
plot_nuclear_populations(ntraj, linestyles,
                         an.datasets["labels"], markers)
plot_total_energies(time, toten, labels)
plot_real_wf(an.datasets["labels"], nstates, krylov_sub_n)
plot_approx_wf(an.datasets["labels"], krylov_sub_n)
plot_approx_el_population(time, an.datasets["labels"], krylov_sub_n, colors)

# Here we use the parameters of a simulation to create a  meaningful directory
# name and copy the output to that directory
if full_H:
    dir_name = "full_" + str(e_gap_thresh) + "_" + str(pop_threshold) + "pop_"\
               + str(nuc_pop_thresh) + "nuccoup_" + str(olapmax) + "olap"
else:
    dir_name = str(krylov_sub_n) + "_" + str(e_gap_thresh) + "_" \
               + str(pop_threshold) + "pop_"\
               + str(nuc_pop_thresh) + "nuccoup_" + str(olapmax) + "olap"
dir_name = dir_name.replace(".", "")

cur_dir = os.getcwd()
path = cur_dir + "/" + dir_name
if not os.path.isdir(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
else:
    print "Directory exists: " + path

# Moving all output files into a directory created for this simulation
filetypes = ("*.dat", "*.hdf5", "*.json", "*.png")
files_to_move = []
for files in filetypes:
    files_to_move.extend(glob.glob(files))

for file_to_move in files_to_move:
    os.system('mv ' + file_to_move + " " + dir_name + "/")

print "Done"
