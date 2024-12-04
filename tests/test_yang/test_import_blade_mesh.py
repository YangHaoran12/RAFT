import capytaine as cpt
from capytaine.io.mesh_loaders import load_VTU, load_STL, load_MSH
import matplotlib.pyplot as plt
# mesh = load_VTU("../models/test00.vtk")
mesh = load_MSH("../models/NREL5MW_rotor.msh")
# mesh.heal_triangles()
mesh.heal_mesh()
mesh.show()


# fig = plt.figure(figsize=(8, 3))
# ax1 = fig.add_subplot(121, projection='3d')
# mesh.show_matplotlib(ax1)

# plt.show()