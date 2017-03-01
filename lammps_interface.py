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
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
if (rank < nprocs // 2 ):
    color = 0
else:
    color = 1
split = comm.Split(color,key=0)

if color == 0:
    from lammps import lammps
    lmp = lammps(name = 'gcc_openmpi', comm = split)
    lmp.command('units metal')
    lmp.command('atom_style atomic')
    lmp.command('dimension 3')
    lmp.command('boundary ss sm pp')
    lmp.command('atom_modify sort 0 0.0 map array')
    lmp.command('read_data cadd_atoms.dat')
    lmp.command('group md_atoms type 1 3 4')
    # lmp.command('group sub_atoms 1 3 4')
    lmp.command('group pad_atoms type 2')
    lmp.command('group interface_atoms type 3')
    lmp.command('group langevin_atoms type 3 4')
    lmp.command('group free_atoms type 1')
    lmp.command('pair_style eam/alloy')
    lmp.command('pair_coeff * * Al_adams_hex.eam.alloy Al Al Al Al Al')
    lmp.command('neighbor 0.1 bin')
    lmp.command('neigh_modify delay 0 every 1 check yes')
    lmp.command('variable mytemp equal 300.0')
    lmp.command('velocity md_atoms create $(2.0*v_mytemp) 829863 dist uniform mom yes rot yes')
    lmp.command("velocity md_atoms set NULL NULL 0.0 units box")
    lmp.command(
        "fix fix_integ md_atoms nve/stadium 300.0 300.0 0.04 699483 stadium -98.249481 96.249481 -98.064113 40.00 -1000.0 1000.00 20.000 ")
    lmp.command('fix fix_zeroforce pad_atoms setforce 0.0 0.0 0.0')
    lmp.command('fix fix_2d all setforce NULL NULL 0.0')
    lmp.command("compute md_temp md_atoms temp/partial 1 1 0")
    lmp.command("compute 1 interface_atoms property/atom id")
    natoms = lmp.get_natoms()
    lmp.command('run 0')
    nlocal = lmp.extract_global('nlocal', 0)
    pp = lmp.extract_compute('1',1,1)
comm.Barrier()

