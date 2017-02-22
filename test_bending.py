from __future__ import print_function
from mpi4py import MPI
import sys
import liboofem
import numpy as np
from lammps import lammps


def initialize_oofem(inputfilename, parallelflag):

    print (inputfilename)
    bcmap = {}
    datareader = liboofem.OOFEMTXTDataReader(inputfilename)
    problem = liboofem.InstanciateProblem(datareader, liboofem.problemMode._processor, 0, parallelflag)
    problem.init()
    problem.checkProblemConsistency()
    problem.postInitialize()
    domain = problem.giveDomain(1)
    nbc = domain.giveNumberOfBoundaryConditions()
    print ("number of boundary Conditions =", nbc)
    for ibc in xrange(nbc):
        bc = domain.giveBc(ibc+1)

    # print ("dofid = ", dofman.giveLabel())
    return datareader, problem, domain

def solve_fem(pb):

    Problem.solveYourselfAt()

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
    if (parallelFlag):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    print (moduleArgs)
    petsc4py.init(moduleArgs)
    from petsc4py import PETSc
    if (parallelFlag):
        inputFilename = inputFilename + "." + str(rank)
        print(rank, inputFilename)

    Datareader, Problem, Domain = initialize_oofem(inputFilename,parallelFlag)
    Datareader.finish()

    Problem.setRenumberFlag()
    print ("problem type = ", type(Problem))
    # Problem.solveYourself()
    smstep = 1
    sjstep = 1
    currentStep = Problem.giveCurrentStep(True)
    nMetaSteps = Problem.giveNumberOfMetaSteps()
    bc1 = Domain.giveBc(1)
    pad_atoms = np.loadtxt('pad_atoms.dat', skiprows=1)
    # ---- Lappms ---

    if (currentStep):
        print ("Current Step is True")
        smstep = Problem.giveMetaStepNumber()
        sjstep = Problem.giveMetaStep(smstep).giveStepRelativeNumber(currentStep.giveNumber()) + 1
    print ("MetaStep", smstep, nMetaSteps)
    # --- Careful with currentStep, in engineering model this is an internal member variable and is set
    for imstep in xrange(smstep, nMetaSteps+1):
        sjstep = 1
        activeMetaStep = Problem.giveMetaStep(imstep)
        Problem.initMetaStepAttributes(activeMetaStep)
        ntimeSteps = activeMetaStep.giveNumberOfSteps()
        print ("MetaStep number = ", smstep, ntimeSteps)
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
            # coord = liboofem.FloatArray(3)
            coordL = [pad_atoms[1,0],pad_atoms[1,1], pad_atoms[1,2]]
            coord = liboofem.FloatArray(3)
            coord[0] = coordL[0]
            e = Domain.giveSpatialLocalizer().giveElementContainingPoint(coord)
            bc1.setPrescribedValue(2.0*jstep)
            print("Applying new BC to ", 2.0*jstep)
            Problem.terminate(curStep)
