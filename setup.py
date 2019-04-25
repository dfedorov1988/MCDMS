from distutils.core import setup

setup(name='pyspawn',
      version='0.0',
      description='Multiple Cloning in Dense Manifolds of States',
      author='Dmitry A. Fedorov, Benjamin G. Levine',
      url='https://github.com/dfedorov1988/MCDMS',
      packages=['pyspawn', 'pyspawn.classical_integrator', 'pyspawn.potential',
                'pyspawn.qm_hamiltonian', 'pyspawn.qm_integrator'])
