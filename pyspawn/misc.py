import numpy as np

def calc_amp_pop(eigenvecs, wave_func, nstates):
    """Calculates amplitudes and population from wave function, eigenvectors"""

    pop = np.zeros(nstates)
    amp = np.zeros((nstates), dtype=np.complex128)
    for j in range(nstates):
        amp[j] = np.dot(eigenvecs[:, j], wave_func)
        pop[j] = np.real(bra_ket(amp[j], amp[j]))

    return amp, pop


def clone_to_a_state(eigenvecs, amp, istate, nstates):
    '''Calculates the child and parent wave functions during
    cloning to a state'''

    child_wf = np.zeros((nstates), dtype=np.complex128)
    parent_wf = np.zeros((nstates), dtype=np.complex128)

    for kstate in range(nstates):
        if kstate == istate:
            # the population is transferred to state i on child
            child_wf += eigenvecs[:, kstate] * amp[kstate]\
                        / np.abs(amp[kstate])

        else:
            # rescaling the rest of the states on the parent function
            parent_wf += eigenvecs[:, kstate] * amp[kstate]\
                       / np.sqrt(1 - bra_ket(amp[istate],
                                             amp[istate]))

    return child_wf, parent_wf


def bra_ket(bra, ket):

    result = np.dot(np.transpose(np.conjugate(bra)), ket)
    return result


def expec_value(vector, operator):
    """Calculates expectation value < bra | operator | ket >"""

    bra = np.transpose(np.conjugate(vector))
    ket = np.dot(operator, vector)
    result = np.dot(bra, ket)

    return result
