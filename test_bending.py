from __future__ import print_function
from mpi4py import MPI
import sys
import liboofem
import numpy as np
from lammps import lammps
# --- Remember extract_compute and extract_fix are all local to each processor and the returned arrays are all local
def initialize_lammps(lmp, comm):
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
    lmp.command("compute 1 all property/atom id")
    natoms = lmp.get_natoms()
    lmp.command('run 0')
    # comm.Barrier()

def update_pad_atom_coords(lmp, pad_atom_new_coords, pad_atom_dict, comm):
    local_atom_ids = lmp.extract_atom('id', 0)
    local_atom_type = lmp.extract_atom('type',0)
    local_atom_coords = lmp.extract_atom('x', 3)
    nlocal = lmp.extract_global('nlocal',0)

    for i in xrange(nlocal):
        if (local_atom_type[i] == 2):
            xid = local_atom_ids[i]
            padid = pad_atom_dict[xid]
            for j in range(3):
                local_atom_coords[i][j] = pad_atom_new_coords[padid][j]


def initialize_oofem(inputfilename, parallelflag, bclist):
    print (inputfilename)
    datareader = liboofem.OOFEMTXTDataReader(inputfilename)
    problem = liboofem.InstanciateProblem(datareader, liboofem.problemMode._processor, 0, parallelflag)
    problem.init()
    problem.checkProblemConsistency()
    problem.postInitialize()
    domain = problem.giveDomain(1)
    nbc = domain.giveNumberOfBoundaryConditions()
    bc1 = domain.giveBc(1)
    setnum = bc1.giveSetNumber()
    print ("Boundary Condition Set = ", setnum)
    bcSet = domain.giveSet(setnum)
    nodeList = bcSet.giveNodeList()
    for i in xrange(len(nodeList)):
        # print (nodeList[i])
        bcList.append(nodeList[i])
    return datareader, problem, domain

def apply_fem_displacement(Domain, bcMap, interface_atom_dict):
    local_atom_ids = lmp.extract_atom('id', 0)
    local_atom_type = lmp.extract_atom('type',0)
    local_dx = lmp.extract_fix('dx_ave', 1,1,1,1)
    local_dy = lmp.extract_fix('dy_ave', 1, 1, 1, 1)
    local_dz = lmp.extract_fix('dz_ave', 1, 1, 1, 1)

    nlocal = lmp.extract_global('nlocal',0)
    bc = Domain.giveBc(1)
    dofs = bc.giveDofIds()
    du = liboofem.DofIDItem.D_u
    dv = liboofem.DofIDItem.D_v
    dw = liboofem.DofIDItem.D_w

    # --- Need Displacements
    for i in xrange(nlocal):
        if (local_atom_type[i] == 3):
            xid = local_atom_ids[i]
            fem_node_num = bcMap[xid]
            dofman = Domain.giveDofManager(fem_node_num)
            for idof in dofs:
                dof = dofman.giveDofWithID(idof)
                if (dof.giveDofId == du):
                    setbc = local_dx[i]
                elif (dof.giveDofId == dv):
                    setbc = local_dy[i]
                elif (dof.giveDofId == dv):
                    setbc = local_dz[i]
                bc.setManualValue(dof,setbc)
            # -- get fem node number and dof a


def get_pad_atom_coords(Domain,pad_atom_coords, pad_atom_new_coords):
    regionlist = None
    i = 0
    for coord in pad_atom_coords:
        spl = Domain.giveSpatialLocalizer()
        elem = spl.giveElementContainingPoint(coord, regionlist)
        lcoord = liboofem.FloatArray(3)
        lcoord.zero()

        err = elem.computeLocalCoordinates(lcoord, coord)
        # ---- Need to Generalize this for different dimenstions
        displacement = liboofem.FloatArray(2)
        displacement.zero()
        vmt = liboofem.ValueModeType.VM_Total
        if (elem is not None):
            elem.computeFiled(vmt, curStep, lcoord, displacement)
            answer = liboofem.FloatArray(3)
            answer = displacement
            answer[2] = 0.0
            pad_atom_new_coords[i].beCopyOf(coord)
            pad_atom_new_coords[i].add(1.0, answer)
        i += 1

if __name__== "__main__":

    parallelFlag = False
    argc = len(sys.argv)
    import petsc4py

    # Parse through arguments
    moduleArgs = []
    i = 1
    if argc != 1:
        moduleArgs.append(sys.argv[0])
    while i < argc:
        if sys.argv[i] == "-f":
            if (i + 1 < argc):
                i += 1
                inputFilename = sys.argv[i]
        elif sys.argv[i] == "-p":
            parallelFlag = True
        elif sys.argv[i] == "-l":
            if i+1 < argc:
                i += 1
                lammpsFilename = sys.argv[i]
        else:
            moduleArgs.append(sys.argv[i])
        i += 1
        # Initialize MPI and petsc
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print (moduleArgs)
    petsc4py.init(moduleArgs)
    from petsc4py import PETSc
    if (parallelFlag):
        inputFilename = inputFilename + "." + str(rank)
        print(rank, inputFilename)



    coords = np.loadtxt('pad_atoms.dat', skiprows=1)
    print ("Shape = ", np.shape(coords))
    numpadatoms = np.shape(coords)[0]
    numcoords = np.shape(coords)[1]
    pad_atom_coords = []
    pad_atom_new_coords = []
    for i in range(numpadatoms):
        c = liboofem.FloatArray(3)
        for j in range(numcoords):
            c[j] = coords[i][j]
        pad_atom_coords.append(c)
        pad_atom_new_coords.append(liboofem.FloatArray(3))
        # print (c)
    del coords

    lmp = lammps(name="gcc_openmpi")
    initialize_lammps(lmp,comm)
    natoms = lmp.get_natoms()
    lmpids = lmp.gather_atoms('id',0,1)
    lmptype = lmp.gather_atoms('type',0,1)

    bcList = []
    Datareader, Problem, Domain = initialize_oofem(inputFilename,parallelFlag, bcList)
    Datareader.finish()

    Problem.setRenumberFlag()
    currentStep = Problem.giveCurrentStep(True)
    nMetaSteps = Problem.giveNumberOfMetaSteps()
    # bc1 = Domain.giveBc(1)

    # ---- Create map between lammps and oofem
    pad_atom_dict = {}
    interface_atom_dict = {}
    bcMap = {}
    l = 0
    k = 0
    for i in xrange(natoms):
        if (lmptype[i] == 2):
            pad_atom_dict[lmpids[i]] = l
            l += 1
        elif (lmptype[i] == 3):
            interface_atom_dict[lmpids[i]] = k
            bcMap[lmpids[i]] = bcList[k]
            k += 1
            # --- Compare coords and create bcMap

    # ---- Lappms ---
    smstep = 1
    sjstep = 1
    if (currentStep):
        smstep = Problem.giveMetaStepNumber()
        sjstep = Problem.giveMetaStep(smstep).giveStepRelativeNumber(currentStep.giveNumber()) + 1
    # --- Careful with currentStep, in engineering model this is an internal member variable and is set
    for imstep in xrange(smstep, nMetaSteps+1):
        sjstep = 1
        activeMetaStep = Problem.giveMetaStep(imstep)
        Problem.initMetaStepAttributes(activeMetaStep)
        ntimeSteps = activeMetaStep.giveNumberOfSteps()
        for jstep in xrange(sjstep, ntimeSteps+1):
            curStep = Problem.giveCurrentStep(True)
            Problem.preInitializeNextStep()
            curStep = Problem.giveNextStep()
            if (Problem.requiresEquationRenumbering):
                Problem.forceEquationNumbering
            Problem.initializeYourself(curStep)
            Problem.solveYourselfAt(curStep)
            # ---- Updates variables at the end of the current solution step
            Problem.updateYourself(curStep)
            # --- Apply boundary conditions for the next step... this is where lammps comes in
            # --- Get positions of pad atoms from fem
            # --- Loop through lammps steps to get interface displacements
            # --- Apply interface displacements
            # bc1.setPrescribedValue(2.0*jstep)
            # print("Applying new BC to ", 2.0*jstep)
            get_pad_atom_coords(Domain,pad_atom_coords,pad_atom_new_coords)
            comm.Barrier()
            # print (pad_atom_new_coords[numpadatoms-1])
            update_pad_atom_coords(lmp, pad_atom_new_coords, pad_atom_dict, comm)
            comm.Barrier()
            apply_fem_displacement(Domain,bcMap, interface_atom_dict)
            comm.Barrier()
            Problem.terminate(curStep)
