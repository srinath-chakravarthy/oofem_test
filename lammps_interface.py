"""
Lammps interface to oofem CADD
Can be generalized
This class should create a lammps object
  be able to gather everything from the lammps object
    displacements
    veclocities
    forces
Contain a map of atom_original positions to some global index and a secondary
  map from global index to oofem index
  perhaps also a third index to convert from this index to processor based index of boundary atoms
The same will be repeated for the pad atoms
It will also contain an interface to a function calls the boundary conditions routine
"""


