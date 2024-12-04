import yaml
from raft.raft_member import Member
from raft.member2Gmsh import memberMeshCirc, writeMesh, memberMeshRect
from raft.raft_mesh import sPlatformMesh

# fname_design ="examples/OC4semi-RAFT_QTF.yaml"
fname_design ="examples/VolturnUS-S_example.yaml"

with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

# dict = design["members"][0]
dict = design["platform"]["members"][2]

mem_list = []
mem1_list = []

mem_list.append(Member(dict, 1,heading=60))
mem_list.append(Member(dict, 1,heading=180))
mem_list.append(Member(dict, 1,heading=300))
dict = design["platform"]["members"][1]
mem_list.append(Member(dict, 1,heading=60))
mem_list.append(Member(dict, 1,heading=180))
mem_list.append(Member(dict, 1,heading=300))
dict = design["platform"]["members"][0]
mem_list.append(Member(dict, 1, heading=0))

for iter, mem in enumerate(mem_list):
    
    mem.setPosition()

aa = sPlatformMesh(mem_list[3:], mem_list[:3])

for iter, mem in enumerate(mem_list):
    
    mem.setPosition()
    memberMeshCirc(mem.stations, mem.d, mem.rA, mem.rB, index=iter+1)

writeMesh(visual=True)