import capytaine as cpt
from capytaine.post_pro.free_surfaces import FreeSurface
from capytaine.bem.airy_waves import airy_waves_free_surface_elevation
from capytaine.ui.vtk import Animation
import numpy as np

# hull_mesh = cpt.Mesh(mesh.getVertices(), mesh.getFaces(), 'OC4')
hull_mesh = cpt.load_mesh('OC4.dat')
hull_mesh.heal_mesh()
from capytaine import Plane
# hull_mesh = hull_mesh.clipped(Plane(point=(0, 0, 0), normal=(0, 1, 0)))

# hull_mesh.show()

cal_mesh, lid_mesh = hull_mesh.extract_lid()
ani_mesh = cpt.load_mesh('OC4_full.dat')
ani_mesh.heal_mesh()
ani_body = cpt.FloatingBody(mesh=ani_mesh, dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, 0)))

# body = cpt.FloatingBody(mesh=cal_mesh, lid_mesh=lid_mesh, dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, 0)))

# import xarray as xr

# body = cpt.FloatingBody(mesh=cal_mesh, lid_mesh=lid_mesh, dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, 0)))

from capytaine.bem.problems_and_results import LinearPotentialFlowProblem, RadiationProblem

body = cpt.FloatingBody(mesh=cal_mesh, dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, -13.46)), center_of_mass = [0,0,-13.46])
body1 = cpt.FloatingBody(mesh=cal_mesh, dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, 0)), center_of_mass = [0,0,-13.46])
# problem = RadiationProblem(radiating_dof="Surge", omega=0.0, body=body)

radiation_problems = [cpt.RadiationProblem(omega=0.0, body=body, radiating_dof=dof, g=9.81, rho=1025) for dof in body.dofs]
radiation_problems.extend([cpt.RadiationProblem(omega=np.inf, body=body, radiating_dof=dof, g=9.81, rho=1025) for dof in body.dofs])

solver = cpt.BEMSolver()

radiation_results = solver.solve_all(radiation_problems, keep_details=True)

dataset = cpt.assemble_dataset(radiation_results)

A0 = dataset['added_mass'].sel(omega=0.0).data
Ainf = dataset['added_mass'].sel(omega=np.inf).data

#######################################################################################################################################################

radiation_problems = [cpt.RadiationProblem(omega=0.0, body=body1, radiating_dof=dof, g=9.81, rho=1025) for dof in body.dofs]
radiation_problems.extend([cpt.RadiationProblem(omega=np.inf, body=body1, radiating_dof=dof, g=9.81, rho=1025) for dof in body.dofs])

solver = cpt.BEMSolver()

radiation_results = solver.solve_all(radiation_problems, keep_details=True)

dataset1 = cpt.assemble_dataset(radiation_results)

A0_ = dataset1['added_mass'].sel(omega=0.0).data
Ainf_ = dataset1['added_mass'].sel(omega=np.inf).data
###########################################################################################################################################################

import matplotlib.pyplot as plt

f = [0,1]

fig,ax = plt.subplots(3,2)
ax[0,0].plot(f, [A0[0,0], Ainf[0,0]])
ax[0,0].plot(f, [A0_[0,0], Ainf_[0,0]])

ax[0,1].plot(f, [A0[1,1], Ainf[1,1]])
ax[0,1].plot(f, [A0_[1,1], Ainf_[1,1]])

ax[1,0].plot(f, [A0[3,3], Ainf[3,3]])
ax[1,0].plot(f, [A0_[3,3], Ainf_[3,3]])

ax[1,1].plot(f, [A0[4,4], Ainf[4,4]])
ax[1,1].plot(f, [A0_[4,4], Ainf_[4,4]])

ax[2,0].plot(f, [A0[5,5], Ainf[5,5]])
ax[2,0].plot(f, [A0_[5,5], Ainf_[5,5]])

ax[0,0].legend(['First line', 'Second line'])

# ax[1,0].plot(f, A44)
# ax[1,0].plot(f, A55)
# ax[1,0].plot(f, A66)

# ax[1,1].plot(f, B44)
# ax[1,1].plot(f, B55)
# ax[1,1].plot(f, B66)

# ax[1,1].plot(f, B44)
# ax[1,1].plot(f, B55)
# ax[1,1].plot(f, B66)

# ax[2,0].plot(f, A24)
# ax[2,0].plot(f, A15)

# ax[2,1].plot(f, B24)
# ax[2,1].plot(f, B15)


plt.show()

