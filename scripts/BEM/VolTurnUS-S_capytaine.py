from raft.raft_bem import CapytaineSolver
from raft.raft_rotor import raft_dir
import os
import numpy as np

# meshFile = os.path.join(raft_dir, 'models/nemoh/OC4_3087.dat')
meshFile = os.path.join(raft_dir, 'models/nemoh/VolTurnUS-S_4159.dat')

solver = CapytaineSolver(memList=None, cog=[0,0,0], include0andinf=True, headings=np.arange(0,210,30), nw=81)
solver.getCapytaineMeshFromFile("models/nemoh/OC4_3087.dat")
solver.defineProblems()

solver.solveProblems(nThreads=8)

solver.postProcess()

solver.plotRadiationResults()
solver.plotFexcResults()

outputDir = os.path.join(raft_dir, 'BEM/VolTurnUS-S-WAMIT')
# outputDir = os.path.join(raft_dir, 'BEM/OC4-semi-WAMIT')

solver.writeWamit(outputDir)
