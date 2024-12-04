import gmsh
import numpy as np
import yaml
import os

# from raft.raft_member import Member

import meshmagick.mmio as mmio
from meshmagick.mesh import Mesh
import meshmagick.hydrostatics as hydro


def memberMeshCirc(stations, diameters, rA, rB, dz_max=0, da_max=0, index=0):

    q = (rB-rA)/stations[-1]
    radii = 0.5*np.array(diameters)
    r_base = rA

    section_list = [] # sections of discontinuous diameters
    gmsh.initialize()

    for i in range(1, len(stations)):

        if stations[i] == stations[i-1]:
            pass

        else:

            dl = stations[i] * q   # the vector of ascending length
            dx,dy,dz = (dl[0], dl[1], dl[2])
            
            if radii[i] != radii[i-1]:
                gmsh.model.occ.add_cone(r_base[0], r_base[1], r_base[2], dx, dy, dz, radii[i-1], radii[i], index+i)

            else:
                gmsh.model.occ.add_cylinder(r_base[0], r_base[1], r_base[2], dx, dy, dz, radii[i-1], index+i)

            r_base += q*stations[i]
            section_list.append((3,index+i))

    if len(section_list) > 1:
        for sec in range (1, len(section_list)):
            gmsh.model.occ.fuse([section_list[0]], [section_list[sec]])

def memberMeshRect(stations, sl, R, dz_max=0, da_max=0, index=0):

    # q = (rB-rA)/stations[-1]
    lx = sl[:,0]
    ly = sl[:,1]
    # r_base = rA
    

    section_list = [] # sections of discontinuous diameters
    gmsh.initialize()

    for i in range(1, len(stations)):

        r_bot_down = np.matmul(R, np.array([-lx[i-1], -ly[i-1], stations[i-1]]))
        r_bot_up =   np.matmul(R, np.array([ lx[i-1],  ly[i-1], stations[i-1]]))
        r_top_down = np.matmul(R, np.array([-lx[i],   -ly[i],   stations[i]]))
        r_top_up   = np.matmul(R, np.array([ lx[i],    ly[i],   stations[i]]))

        dx = r_top_up[0] - r_bot_down[0]
        dy = r_top_up[1] - r_bot_down[1]
        dz = r_top_up[2] - r_bot_down[2]

        if stations[i] == stations[i-1]:
            pass

        else:

            # dl = stations[i] * q   # the vector of ascending length
            # dx,dy,dz = (dl[0], dl[1], dl[2])
            
            if lx[i] != lx[i-1] or ly[i] != ly[i-1]:

                dx = r_bot_up[0] - r_bot_down[0]
                dy = r_bot_up[1] - r_bot_down[1]
                dz = r_bot_up[2] - r_bot_down[2]
                # raise Exception("member with a changing side length is not implemented yet")
                gmsh.model.occ.addRectangle(r_bot_down[0], r_bot_down[1], r_bot_down[2], dx, dy, index+i)
                # gmsh.model.occ.addCurveLoop([1], index+i)

                dx = r_top_up[0] - r_top_down[0]
                dy = r_top_up[1] - r_top_down[1]
                dz = r_top_up[2] - r_top_down[2]
                gmsh.model.occ.addRectangle(r_top_down[0], r_top_down[1], r_top_down[2], dx, dy, index+i+1)
                # gmsh.model.occ.addCurveLoop([1], index+i+1)

                gmsh.model.occ.addThruSections([index+i, index+i+1], index+i)


            else:

                dx = r_top_up[0] - r_bot_down[0]
                dy = r_top_up[1] - r_bot_down[1]
                dz = r_top_up[2] - r_bot_down[2]

                gmsh.model.occ.addBox(r_bot_down[0], r_bot_down[1], r_bot_down[2], dx, dy, dz, index+i)


            # r_base += q*stations[i]
            section_list.append((3,index+i))

    if len(section_list) > 1:
        for sec in range (1, len(section_list)):
            gmsh.model.occ.fuse([section_list[0]], [section_list[sec]])



def writeMesh(dir="LHEEA", meshFile="mesh.msh", visual=False, verbose=True, hydroFlag=True):

    if not os.path.isdir(dir):
        os.mkdir(dir)

    gmsh.model.occ.synchronize()

    surf_list = gmsh.model.getEntities(2)
    surf = [i[1] for i in surf_list]
    gmsh.model.addPhysicalGroup(2, surf, 2, "surface")

    gmsh.option.setNumber("Mesh.MeshSizeMin", 1)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 1)

    gmsh.option.setNumber('Mesh.Format', 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.SaveParametric", 0)

    gmsh.model.mesh.generate(2)

    gmsh.model.occ.synchronize()
    nodes, coords, _ = gmsh.model.mesh.getNodes()
    _, elements, node_list = gmsh.model.mesh.getElements()

    vertices = np.array(coords).reshape(len(nodes),3)
    faces = np.array(node_list[1]-1).reshape(len(elements[1]),3)
    faces = np.hstack((faces, faces[:, [0]]))

    output_file = "LHEEA/mem_subm_horz_rect_cyl.msh"
    vertices1, faces1 = mmio.load_MSH22(output_file)

    mesh = Mesh(vertices, faces, "tmp")
    mesh.quick_save()

    if visual is True:
        gmsh.fltk.run()
    
    ouputFile = os.path.join(dir, meshFile)
    gmsh.write(ouputFile)

    # def compute_hydrostatics():
    #     vertices, faces = mmio.load_MSH22(ouputFile)
    #     obj = Mesh(vertices, faces)

    #     hs_data = hydro.compute_hydrostatics(obj, [0,0,0], 1025, 9.81)
    #     print(hydro.get_hydrostatic_report(hs_data))

    # if hydroFlag is True:
    #     compute_hydrostatics() 




###########################################################
# fname_design ="examples/OC4semi-RAFT_QTF.yaml"

# with open(fname_design) as file:
#     design = yaml.load(file, Loader=yaml.FullLoader)

# # dict = design["members"][0]
# dict = design["platform"]["members"][1]

# mem_list = []

# mem_list.append(Member(dict, 1,heading=60))
# mem_list.append(Member(dict, 1,heading=180))
# mem_list.append(Member(dict, 1,heading=300))

# for iter, mem in enumerate(mem_list):
#     mem.setPosition()
#     memberMesh(mem.stations, mem.d, mem.rA, mem.rB, index=iter+1)

# writeMesh(visual=True)
##########################################################
# gmsh.model.occ.synchronize()
# surf_ = gmsh.model.getEntities(2)
# surf = [i[1] for i in surf_]
# gmsh.model.addPhysicalGroup(2, surf, 2, "surface")



##########################################################
# rA = [0,0,0]
# rB = [0,0,10]
# rC = [0,0,20]

# l = 20
# radii = [5, 4, 3]

# q = np.array([0,0,1])
# stations = np.array([0, 0.5, 1])*l

# gmsh.initialize()

# i = 1
# dr = q * (stations[i] - stations[i-1])
# dx = dr[0]
# dy = dr[1]
# dz = dr[2]

# gmsh.model.occ.add_cone(rA[0], rA[1], rA[2], dx, dy, dz, radii[i-1], radii[i], 1)

# s0 = gmsh.model.occ.get_entities(dim=2)

# i = 2
# dr = q * (stations[i] - stations[i-1])
# dx = dr[0]
# dy = dr[1]
# dz = dr[2]

# gmsh.model.occ.add_cone(rB[0], rB[1], rB[2], dx, dy, dz, radii[i]+0.1, radii[i], 2)

# # s1 = gmsh.model.occ.get_entities(dim=2)
# # sdiff = []
# # for i in s1:
# #     if i not in s0:
# #         sdiff.append(i)
# # print(sdiff)

# # cut = gmsh.model.occ.cut(s1, sdiff)
# # print(cut)
# # gmsh.model.occ.remove([(2,6)])

# gmsh.model.occ.fuse([(3,1)], [(3,2)])
# inter = gmsh.model.occ.intersect([(3,1)], [(3,2)], removeObject=False, removeTool=False)
# print(inter)
# s = gmsh.model.occ.get_entities(dim= 2)
# print(s)

# gmsh.option.setNumber("Mesh.MeshSizeMin", 3)
# gmsh.option.setNumber("Mesh.MeshSizeMax", 3)

# gmsh.option.setNumber('Mesh.Format', 1)
# gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
# gmsh.option.setNumber("Mesh.SaveParametric", 0)
# gmsh.option.setNumber("Mesh.SaveParametric", 0)

# gmsh.model.mesh.generate(2)

# outputt_file = "tests/OC4.msh"
# gmsh.write(outputt_file)
# # node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
# # # face_tags, face_orient    = gmsh.model.mesh.getAllFaces(faceType=3)
# # _, elemen, nodes = gmsh.model.mesh.getElements(2)

# gmsh.fltk.run()
# gmsh.finalize()

# vertices, faces = mmio.load_MSH22(outputt_file)
# obj = Mesh(vertices, faces)

# hs_data = compute_hydrostatics(obj, [0,0,-13.46], 1025, 9.81)
# print(get_hydrostatic_report(hs_data))

# a = 1


