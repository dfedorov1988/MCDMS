import os


def print_splash():
    """Prints splash screen"""

    print "\nYou are about to propagate a molecular wave function with MCDMS"
    print "\n------------------------------------------------"
    print "Multiple Cloning in Dense Manifolds of States"
    print "Nonadiabatic molecular dynamics software package\n" +\
          "written by Benjamin G. Levine, Dmitry A. Fedorov"
    print "------------------------------------------------"


def check_files():
    print " Checking for files from previous run"
    if (os.path.isfile("working.hdf5")) or (os.path.isfile("sim.hdf5")):
        print "! working.hdf5 is present."
        erase_hdf5 = raw_input("Would you like to delete hdf5 files?('y')\n")
        if erase_hdf5 == "y":
            os.system("rm *.hdf5")
        else:
            quit()
