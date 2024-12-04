from raft.raft_bem import CapytaineSolver, animation
from raft.raft_rotor_yang import raft_dir
import numpy as np
import os





animation(calMeshFile=os.path.join(raft_dir, 'BEM/VolturnUS-S/platform.dat'),
          platformMeshFile=os.path.join(raft_dir, 'models/nemoh/VolturnUS-S_platform.dat'),
          rotorMeshFile=os.path.join(raft_dir, 'models/nemoh/IEA-15MW-rotor.dat'),
          towerMeshFile=os.path.join(raft_dir, 'models/nemoh/VolturnUS-S_tower.dat'),
          rao = [0.1,0,0,0,0.01,0],
          omega=0.25*6.18,
          eta=2,
          heading=30
)