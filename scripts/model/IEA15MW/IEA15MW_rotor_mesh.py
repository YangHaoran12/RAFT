import yaml
from raft.raft_rotor_yang import Rotor, raft_dir
import numpy as np
import os

fname_design = os.path.join(raft_dir, "examples/VolturnUS-S_example.yaml")
# fname_design ="examples/OC4semi-RAFT_QTF.yaml"

with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

    design['turbine']['rho_air' ] = design['site']['rho_air']
    design['turbine']['mu_air'  ] = design['site']['mu_air']
    design['turbine']['shearExp_air'] = design['site']['shearExp']
    design['turbine']['shearExp_water'] = design['site']['shearExp']

    # zero the nacelle velocity feedback gain since there seems to be a discrepancy with its definition
    design['turbine']['pitch_control']['Fl_Kp'] = 0.0


rotor_dict = design["turbine"]
rotor = Rotor(rotor_dict, [1], 0)
rotor.setPosition()

geometry_table = np.array(design["turbine"]['blade'][0]['geometry'])

meshFile = os.path.join(raft_dir, "models/msh/IEA-15MW-rotor.msh")

# geometry_table[:, 3] = -geometry_table[:, 3]

rotor.baldeGeo2Mesh(geometry_table, 
                    show=True, 
                    all=True, 
                    mesh=True, 
                    vtk_show=True, 
                    meshFile=meshFile, 
                    shroud_radius=6, 
                    save=True,
                    constraint=0.05)

