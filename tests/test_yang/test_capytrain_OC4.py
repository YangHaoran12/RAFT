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

body = cpt.FloatingBody(mesh=cal_mesh, dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, -13.46)), center_of_mass = cal_mesh.center_of_buoyancy)

# problem = RadiationProblem(radiating_dof="Surge", omega=0.0, body=body)
radiation_problems = []
# radiation_problems = [cpt.RadiationProblem(omega=0.0, body=body, radiating_dof=dof, g=9.81, rho=1025) for dof in body.dofs]
radiation_problems.extend([cpt.RadiationProblem(omega=np.inf, body=body, radiating_dof=dof, g=9.81, rho=1025) for dof in body.dofs])

solver = cpt.BEMSolver()

radiation_results = solver.solve_all(radiation_problems, keep_details=True)

print(radiation_results[0].radiation_damping)

dataset = cpt.assemble_dataset(radiation_results)

dataset.to_dataframe().to_csv('tmp1.csv')


a = 1

# body = cpt.FloatingBody(mesh=cal_mesh, lid_mesh=lid_mesh, dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, 0)))

# fs = FreeSurface(x_range=(-50.0, 50.0), nx=100, y_range=(-50.0, 50.0), ny=100)
# problem = LinearPotentialFlowProblem(body=body, water_depth=200, omega=0.628, rho=1025, wave_direction=0)

# solver = cpt.BEMSolver()
# results = solver.solve(problem, keep_details=True)

# a = 1
import xarray as xr

body = cpt.FloatingBody(mesh=cal_mesh, lid_mesh=lid_mesh, dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, -13.46)), center_of_mass = cal_mesh.center_of_buoyancy)
body.inertia_matrix = body.compute_rigid_body_inertia() # Artificially lower to have a more appealing animation
body.hydrostatic_stiffness = body.compute_hydrostatic_stiffness()


test_matrix = xr.Dataset(coords={

    'omega': 0.628,
    'wave_direction': np.linspace(0, np.pi/2, 2),
    'radiating_dof': list(body.dofs),
    'water_depth': [np.inf],
})

# test_matrix = xr.Dataset(coords={

#     'omega': np.linspace(0.0628, 0.628*8, 81),
#     'rho'  : 1025,
#     'wave_direction': 0,
#     'radiating_dof': list(body.dofs),
#     'water_depth': [np.inf],
# })
# result = cpt.BEMSolver().solve(test_matrix)
dataset = cpt.BEMSolver().fill_dataset(test_matrix, body, n_jobs=4)

A = dataset['added_mass'].sel(omega=0.628)

# A11 = dataset['added_mass']
# print(A11)
# print(A11.data)

A11 = dataset['added_mass'].sel(radiating_dof='Surge', influenced_dof='Surge')
A22 = dataset['added_mass'].sel(radiating_dof='Sway', influenced_dof='Sway')
A33 = dataset['added_mass'].sel(radiating_dof='Heave', influenced_dof='Heave')
A44 = dataset['added_mass'].sel(radiating_dof='Roll', influenced_dof='Roll')
A55 = dataset['added_mass'].sel(radiating_dof='Pitch', influenced_dof='Pitch')
A66 = dataset['added_mass'].sel(radiating_dof='Yaw', influenced_dof='Yaw')

A15 = dataset['added_mass'].sel(radiating_dof='Surge', influenced_dof='Pitch')
A24 = dataset['added_mass'].sel(radiating_dof='Sway', influenced_dof='Roll')

B11 = dataset['radiation_damping'].sel(radiating_dof='Surge', influenced_dof='Surge')
B22 = dataset['radiation_damping'].sel(radiating_dof='Sway', influenced_dof='Sway')
B33 = dataset['radiation_damping'].sel(radiating_dof='Heave', influenced_dof='Heave')
B44 = dataset['radiation_damping'].sel(radiating_dof='Roll', influenced_dof='Roll')
B55 = dataset['radiation_damping'].sel(radiating_dof='Pitch', influenced_dof='Pitch')
B66 = dataset['radiation_damping'].sel(radiating_dof='Yaw', influenced_dof='Yaw')

B15 = dataset['radiation_damping'].sel(radiating_dof='Surge', influenced_dof='Pitch')
B24 = dataset['radiation_damping'].sel(radiating_dof='Sway', influenced_dof='Roll')

import matplotlib.pyplot as plt

f = np.linspace(0.01, 0.8, 81)

fig,ax = plt.subplots(3,2)
ax[0,0].plot(f, A11)
ax[0,0].plot(f, A22)
ax[0,0].plot(f, A33)

ax[0,1].plot(f, B11)
ax[0,1].plot(f, B22)
ax[0,1].plot(f, B33)

ax[1,0].plot(f, A44)
ax[1,0].plot(f, A55)
ax[1,0].plot(f, A66)

ax[1,1].plot(f, B44)
ax[1,1].plot(f, B55)
ax[1,1].plot(f, B66)

ax[1,1].plot(f, B44)
ax[1,1].plot(f, B55)
ax[1,1].plot(f, B66)

ax[2,0].plot(f, A24)
ax[2,0].plot(f, A15)

ax[2,1].plot(f, B24)
ax[2,1].plot(f, B15)


plt.show()
#######################################################################################
# body = cpt.FloatingBody(mesh=cal_mesh, lid_mesh=lid_mesh, dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, 0)), center_of_mass = cal_mesh.center_of_buoyancy)
# body.inertia_matrix = body.compute_rigid_body_inertia() # Artificially lower to have a more appealing animation
# body.hydrostatic_stiffness = body.compute_hydrostatic_stiffness()

# # body.show()

# # body.inertia_matrix = body.add_dofs_labels_to_matrix(np.zeros((6,6)))
# #######################################################################################
# import xarray as xr
# test_matrix = xr.Dataset(coords={
#     'omega': [1.0, 2.0],
#     'wave_direction': [0, np.pi/2],
#     'radiating_dof': list(body.dofs),
#     'water_depth': [np.inf],
# })

# dataset = cpt.BEMSolver().fill_dataset(test_matrix, body).to_dataarray()

# problem = cpt.RadiationProblem(body=body, omega=np.inf, radiating_dof="Heave")
# solver = cpt.BEMSolver()
# result = solver.solve(problem, keep_details=True)

# dataset = cpt.BEMSolver().fill_dataset(test_matrix, body)

# a = 1


# #######################################################################################
# fs = FreeSurface(x_range=(-100.0, 100.0), nx=100, y_range=(-100.0, 100.0), ny=100)
# f_hz = 1/12
# omega = f_hz*np.pi*2
# wave_direction = np.pi/4
# wave_amplitude = 6

# solver = cpt.BEMSolver()

# radiation_problems = [cpt.RadiationProblem(omega=0.0, body=body, radiating_dof=dof) for dof in body.dofs]
# radiation_results = solver.solve_all(radiation_problems)
# diffraction_problem = cpt.DiffractionProblem(omega=omega, body=body, wave_direction=wave_direction)
# diffraction_result = solver.solve(diffraction_problem)

# dataset = cpt.assemble_dataset(radiation_results + [diffraction_result])
# rao = cpt.post_pro.rao(dataset, wave_direction=wave_direction)

# incoming_waves_elevation = airy_waves_free_surface_elevation(fs, diffraction_result)
# diffraction_elevation = solver.compute_free_surface_elevation(fs, diffraction_result)

# radiation_elevations_per_dof = {res.radiating_dof: solver.compute_free_surface_elevation(fs, res) for res in radiation_results}
# radiation_elevation = sum(rao.sel(omega=omega, radiating_dof=dof).data * radiation_elevations_per_dof[dof] for dof in body.dofs)

# rao_faces_motion = sum(rao.sel(omega=omega, radiating_dof=dof).data * ani_body.dofs[dof] for dof in ani_body.dofs)

# animation = Animation(loop_duration=2*np.pi/omega)
# animation.add_body(ani_body, faces_motion=wave_amplitude*rao_faces_motion)
# animation.add_free_surface(fs, wave_amplitude * (incoming_waves_elevation + diffraction_elevation + radiation_elevation))

# animation.run(camera_position=(-200,-200,200))

# # fse = solver.get_free_surface_elevation(result=results, free_surface=ws)

# # anim = body_anim.animate(motion={"Surge": 0j}, loop_duration=1.0)
# # anim.add_free_surface(ws, fse+incoming_fse)
# # anim.run(camera_position=(-200,-200,200))

# animation.save("tmp.ogv",camera_position=(-200,-200,200), nb_loops=10)



# a = 1