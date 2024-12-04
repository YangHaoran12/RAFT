import numpy as np
import yaml
import os
import copy

from raft.raft_member import Member

# from raft.raft_member import Member


class MemberMesh(object):

    def __init__(self, mem=Member, meshMode=1, sizeMin=0.1, sizeMax=1.0, constraint=0.25, recombined=False, structured=False, clip=False):
        '''Initialize a MemberMesh object.

        PARAMETERS
        ----------

        mem : raft_member.Member
            A Member object initialized and position has been set
        nw : int
            Number of frequencies in the analysis - used for initializing.
        heading : float, optional
            Heading rotation to apply to the coordinates when setting up the 
            member. Used for member arrangements or FOWT heading offsets [deg].

        '''
        mem = copy.deepcopy(mem)

        if meshMode == 0:

            if mem.shape == "rectangular":
                print("structural method for rect member has not been implemented yet, change to Gmsh method") 
                self._vertices, self._faces = self.__gmshRectMember(stations=mem.stations,
                                                                    sl=mem.sl,
                                                                    rA=mem.rA,
                                                                    rB=mem.rB, 
                                                                    R=mem.R,
                                                                    sizeMin=sizeMin,
                                                                    sizeMax=sizeMax,
                                                                    constraint=constraint,
                                                                    recombined=recombined,
                                                                    structured=structured,
                                                                    clip=clip)

            elif mem.shape == "circular":
                self._vertices, self._faces = self.__strucCircMember(mem.stations, mem.d, mem.rA, mem.rB)

            else:
                raise ValueError(f"Member.shape should be 'rectangular' or 'circular', {mem.shape} was provided instead")

        if meshMode == 1:
            
            if mem.shape == "rectangular":
                self._vertices, self._faces = self.__gmshRectMember(stations=mem.stations,
                                                                    sl=mem.sl,
                                                                    rA=mem.rA,
                                                                    rB=mem.rB, 
                                                                    R=mem.R,
                                                                    sizeMin=sizeMin,
                                                                    sizeMax=sizeMax,
                                                                    constraint=constraint,
                                                                    recombined=recombined,
                                                                    structured=structured,
                                                                    clip=clip)
            
            elif mem.shape == "circular":
                self._vertices, self._faces = self.__gmshCircMember(stations=mem.stations,
                                                                    diameters=mem.d,
                                                                    rA=mem.rA,
                                                                    rB=mem.rB, 
                                                                    sizeMin=sizeMin,
                                                                    sizeMax=sizeMax,
                                                                    constraint=constraint,
                                                                    recombined=recombined,
                                                                    structured=structured,
                                                                    clip=clip)
            
            else:
                raise ValueError(f"Member.shape should be 'rectangular' or 'circular', {mem.shape} was provided instead")

    def __gmshCircMember(self, stations, diameters, rA, rB, sizeMin=0.1, sizeMax=1, constraint=0.25, recombined=True, structured=True, clip=True):
        '''Circular member meshing based on Gmsh.

        PARAMETERS
        ----------

        member : raft_member.Member
            A Member object initialized and position has been set
        nw : int
            Number of frequencies in the analysis - used for initializing.
        heading : float, optional
            Heading rotation to apply to the coordinates when setting up the 
            member. Used for member arrangements or FOWT heading offsets [deg].

        '''
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal",0)

        if rA[2]*rB[2] < 0:
            immersed = True
        else:
            immersed = False

        q = (rB-rA)/stations[-1]
        radii = 0.5*np.array(diameters)
        rOrig = rA

        sections = [] # sections of discontinuous diameters

        for i in range(1, len(stations)):

            if stations[i] == stations[i-1]:
                pass

            else:

                dl = (stations[i]-stations[i-1]) * q   # the vector of ascending length
                dx,dy,dz = (dl[0], dl[1], dl[2])

                if radii[i] != radii[i-1]:
                    gmsh.model.occ.addCone(rOrig[0], rOrig[1], rOrig[2], dx, dy, dz, radii[i-1], radii[i], i)

                else:
                    gmsh.model.occ.addCylinder(rOrig[0], rOrig[1], rOrig[2], dx, dy, dz, radii[i-1], i)

                rOrig += q*(stations[i]-stations[i-1]) 
                sections.append((3,i))

        if len(sections) > 1:
            gmsh.model.occ.fuse([sections[0]], sections[1:])
        
        gmsh.model.occ.synchronize()
        # gmsh.fltk.run()


        if immersed is True and clip is True:
            r = gmsh.model.occ.getBoundingBox(3,1)
            r0 = np.array([r[0],r[1]])
            r1 = np.array([r[3],r[4]])
            rc = (r1+r0) * 0.5
            rmax = np.linalg.norm(r1-rc)
            cutCylinder = gmsh.model.occ.addCylinder(rc[0],rc[1], 0, 0, 0, 500, rmax*1.25)
            # gmsh.model.occ.synchronize()
            # gmsh.fltk.run()
            gmsh.model.occ.cut([(3,1)], [(3,cutCylinder)])
            # plane = gmsh.model.occ.getE
            # gmsh.model.occ.remove([(2, cutPlane), (3,1)], recursive=True)
            # gmsh.model.occ.removeAllDuplicates()
            # gmsh.model.occ.synchronize()
            # gmsh.fltk.run()
        
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", sizeMin)
        gmsh.option.setNumber("Mesh.MeshSizeMax", sizeMax)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), np.min(diameters)*constraint)
        gmsh.option.setNumber('Mesh.Format', 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.SaveParametric", 0)
        gmsh.option.setNumber("Mesh.AngleSmoothNormals", 15)
        gmsh.option.setNumber('Mesh.Smoothing', 5)
        gmsh.option.setNumber('Mesh.SmoothNormals', 1)

        if structured and recombined:
            gmsh.option.setNumber('Mesh.Algorithm', 11)
        elif structured:
            gmsh.option.setNumber('Mesh.Algorithm', 11)
        elif recombined:
            gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)
            gmsh.option.setNumber('Mesh.RecombineAll', 1)

        gmsh.model.mesh.generate(2)     # All triangular elements

        nodes, coords, _ = gmsh.model.mesh.getNodes()
        elementType, elements, elementNodes = gmsh.model.mesh.getElements()

        # gmsh.fltk.run()
        
        gmsh.finalize()

        nVertices = len(nodes)
        vertices = np.array(coords).reshape(nVertices, 3)

        if len(elementType[1:-1]) == 2: # triangular and quadature faces
            nTriFaces    = elements[1].shape[0]
            nQuadFaces   = elements[2].shape[0]

            faces_ = np.array(elementNodes[1]-1).reshape(nTriFaces ,3)
            triFaces = np.hstack((faces_, faces_[:, [0]]))

            quadFaces = np.array(elementNodes[2]-1).reshape(nQuadFaces ,4)

            faces = np.asarray(np.vstack((triFaces, quadFaces)))

        elif len(elementType[1:-1]) == 1 and elementType[1] == 2: # all triangular faces
            nFaces = elements[1].shape[0]

            faces_ = np.array(elementNodes[1]-1).reshape(nFaces ,3)
            faces = np.hstack((faces_, faces_[:, [0]]))

        elif len(elementType[1:-1]) == 1 and elementType[1] == 3: # all rectangular faces
            nFaces = elements[1].shape[0]

            faces = np.array(elementNodes[1]-1).reshape(nFaces ,4)

        return vertices, faces

    def __gmshEllipseMember(self, stations, sl, rA, rB, sizeMin=0.1, sizeMax=1, constraint=0.25, recombined=True, structured=True, clip=True):
        '''Circular member meshing based on Gmsh.

        PARAMETERS
        ----------

        member : raft_member.Member
            A Member object initialized and position has been set
        nw : int
            Number of frequencies in the analysis - used for initializing.
        heading : float, optional
            Heading rotation to apply to the coordinates when setting up the 
            member. Used for member arrangements or FOWT heading offsets [deg].

        '''
        import gmsh
        gmsh.initialize()

        if rA[2]*rB[2] < 0:
            immersed = True
        else:
            immersed = False

        q = (rB-rA)/stations[-1]
        rOrig = rA

        sections = [] # sections of discontinuous diameters

        for i in range(1, len(stations)):

            if stations[i] == stations[i-1]:
                pass

            else:

                # dl = stations[i] * q   # the vector of ascending length
                # dx,dy,dz = (dl[0], dl[1], dl[2])
                
                p = np.matmul(np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]]), q) # x-axis to define the longer side
                disk0 = gmsh.model.occ.addDisk(rOrig[0], rOrig[1], rOrig[2], sl[0,0], sl[1,0], q, p)
                
                rOrig += q*stations[i]
                disk1 = gmsh.model.occ.addDisk(rOrig[0], rOrig[1], rOrig[2], sl[0,0], sl[1,0], q, p)

                gmsh.model.occ.addThruSections([(2, disk0), (2, disk1)], i)

                sections.append((3,i))

        if len(sections) > 1:
            for sec in range (1, len(sections)):
                gmsh.model.occ.fuse([sections[0]], [sections[sec]])


        if immersed is True and clip is True:
            r = gmsh.model.occ.getBoundingBox(3,1)
            r0 = np.array([r[0],r[1]])
            r1 = np.array([r[3],r[4]])
            rc = (r1+r0) * 0.5
            rmax = np.linalg.norm(r1-rc)
            cutCylinder = gmsh.model.occ.addCylinder(rc[0],rc[1], 0, 0, 0, 200, rmax*1.25)
            # gmsh.model.occ.synchronize()
            # gmsh.fltk.run()
            gmsh.model.occ.cut([(3,1)], [(3,cutCylinder)])
            # plane = gmsh.model.occ.getE
            # gmsh.model.occ.remove([(2, cutPlane), (3,1)], recursive=True)
            # gmsh.model.occ.removeAllDuplicates()
            # gmsh.model.occ.synchronize()
            # gmsh.fltk.run()
        
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", sizeMin)
        gmsh.option.setNumber("Mesh.MeshSizeMax", sizeMax)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), np.min(sl[1,:])*constraint)
        gmsh.option.setNumber('Mesh.Format', 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.SaveParametric", 0)
        gmsh.option.setNumber("Mesh.AngleSmoothNormals", 15)
        gmsh.option.setNumber('Mesh.Smoothing', 5)
        gmsh.option.setNumber('Mesh.SmoothNormals', 1)

        if structured and recombined:
            gmsh.option.setNumber('Mesh.Algorithm', 11)
        elif structured:
            gmsh.option.setNumber('Mesh.Algorithm', 11)
        elif recombined:
            gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)
            gmsh.option.setNumber('Mesh.RecombineAll', 1)

        gmsh.model.mesh.generate(2)     # All triangular elements

        nodes, coords, _ = gmsh.model.mesh.getNodes()
        elementType, elements, elementNodes = gmsh.model.mesh.getElements()

        # gmsh.fltk.run()
        
        gmsh.finalize()

        nVertices = len(nodes)
        vertices = np.array(coords).reshape(nVertices, 3)

        if len(elementType[1:-1]) == 2: # triangular and quadature faces
            nTriFaces    = elements[1].shape[0]
            nQuadFaces   = elements[2].shape[0]

            faces_ = np.array(elementNodes[1]-1).reshape(nTriFaces ,3)
            triFaces = np.hstack((faces_, faces_[:, [0]]))

            quadFaces = np.array(elementNodes[2]-1).reshape(nQuadFaces ,4)

            faces = np.asarray(np.vstack((triFaces, quadFaces)))

        elif len(elementType[1:-1]) == 1 and elementType[1] == 2: # all triangular faces
            nFaces = elements[1].shape[0]

            faces_ = np.array(elementNodes[1]-1).reshape(nFaces ,3)
            faces = np.hstack((faces_, faces_[:, [0]]))

        elif len(elementType[1:-1]) == 1 and elementType[1] == 3: # all rectangular faces
            nFaces = elements[1].shape[0]

            faces = np.array(elementNodes[1]-1).reshape(nFaces ,4)

        return vertices, faces

    def __gmshRectMember(self, stations, sl, rA, rB, R, sizeMin=0.1, sizeMax=1, constraint=0.25, recombined=True, structured=True, clip=True):
        '''Rectangular member meshing based on Gmsh.

        PARAMETERS
        ----------

        member : raft_member.Member
            A Member object initialized and position has been set
        nw : int
            Number of frequencies in the analysis - used for initializing.
        heading : float, optional
            Heading rotation to apply to the coordinates when setting up the 
            member. Used for member arrangements or FOWT heading offsets [deg].

        '''
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal",0)
        
        lx = 0.5*sl[:,0]
        ly = 0.5*sl[:,1]

        if rA[2]*rB[2] < 0:
            immersed = True
        else:
            immersed = False

        sections = [] # sections of discontinuous diameters
        index = 1   # indicator of meshing numbering

        for i in range(1, len(stations)):

            rBot = np.array([[-lx[i-1], -ly[i-1], stations[i-1]],
                             [ lx[i-1], -ly[i-1], stations[i-1]],
                             [ lx[i-1],  ly[i-1], stations[i-1]],
                             [-lx[i-1],  ly[i-1], stations[i-1]]]).T
            rTop = np.array([[-lx[i],   -ly[i],   stations[i]],
                             [ lx[i],   -ly[i],   stations[i]],
                             [ lx[i],    ly[i],   stations[i]],
                             [-lx[i],    ly[i],   stations[i]]]).T
            
            rBot = np.matmul(R, rBot) + np.array([[rA[0]], [rA[1]], [rA[2]]])
            rTop = np.matmul(R, rTop) + np.array([[rA[0]], [rA[1]], [rA[2]]])

            if stations[i] == stations[i-1]:
                pass

            else:
                ptsBot = []
                ptsTop = []

                for i in range(4):
                    ptsBot.append(gmsh.model.occ.addPoint(rBot[0, i], rBot[1, i], rBot[2, i], tag=index+i))
                    ptsTop.append(gmsh.model.occ.addPoint(rTop[0, i], rTop[1, i], rTop[2, i], tag=index+i+4))

                for j in range(3):
                    gmsh.model.occ.addLine(ptsBot[j], ptsBot[j+1], j+index)
                    gmsh.model.occ.addLine(ptsTop[j], ptsTop[j+1], j+index+4)

                gmsh.model.occ.addLine(ptsBot[-1], ptsBot[0], index+3)
                gmsh.model.occ.addLine(ptsTop[-1], ptsTop[0], index+7)

                gmsh.model.occ.addCurveLoop([1,2,3,4], index)
                gmsh.model.occ.addCurveLoop([5,6,7,8], index+1)

                gmsh.model.occ.addThruSections([index, index+1], index)

                sections.append((3,index))

                index += 2

        if len(sections) > 1:
            gmsh.model.occ.fuse([sections[0]], [sections[1:]])
        gmsh.model.occ.synchronize()

        if immersed is True and clip is True:
            r = gmsh.model.occ.getBoundingBox(3,1)
            r0 = np.array([r[0],r[1]])
            r1 = np.array([r[3],r[4]])
            rc = (r1+r0) * 0.5
            rmax = np.linalg.norm(r1-rc)
            cutCylinder = gmsh.model.occ.addCylinder(rc[0],rc[1], 0, 0, 0, 500, rmax*1.25)
            gmsh.model.occ.cut([(3,1)], [(3,cutCylinder)])

        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", sizeMin)
        gmsh.option.setNumber("Mesh.MeshSizeMax", sizeMax)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), np.min(sl)*constraint)
        gmsh.option.setNumber('Mesh.Format', 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.SaveParametric", 0)
        gmsh.option.setNumber("Mesh.AngleSmoothNormals", 8)

        if structured and recombined:
            gmsh.option.setNumber('Mesh.Algorithm', 11)
        elif structured:
            gmsh.option.setNumber('Mesh.Algorithm', 11)
        elif recombined:
            gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)
            gmsh.option.setNumber('Mesh.RecombineAll', 1)

        gmsh.model.mesh.generate(2)     # All triangular elements

        nodes, coords, _ = gmsh.model.mesh.getNodes()
        nodes, coords, _ = gmsh.model.mesh.getNodes()
        elementType, elements, elementNodes = gmsh.model.mesh.getElements()

        # gmsh.fltk.run()
        
        gmsh.finalize()

        nVertices = len(nodes)
        vertices = np.array(coords).reshape(nVertices, 3)

        if len(elementType[1:-1]) == 2: # triangular and quadature faces
            nTriFaces    = elements[1].shape[0]
            nQuadFaces   = elements[2].shape[0]

            faces_ = np.array(elementNodes[1]-1).reshape(nTriFaces ,3)
            triFaces = np.hstack((faces_, faces_[:, [0]]))

            quadFaces = np.array(elementNodes[2]-1).reshape(nQuadFaces ,4)

            faces = np.asarray(np.vstack((triFaces, quadFaces)))

        elif len(elementType[1:-1]) == 1 and elementType[1] == 2: # all triangular faces
            nFaces = elements[1].shape[0]

            faces_ = np.array(elementNodes[1]-1).reshape(nFaces ,3)
            faces = np.hstack((faces_, faces_[:, [0]]))

        elif len(elementType[1:-1]) == 1 and elementType[1] == 3: # all rectangular faces
            nFaces = elements[1].shape[0]

            faces = np.array(elementNodes[1]-1).reshape(nFaces ,4)

        return vertices, faces

    def __strucCircMember(self, stations, diameters, rA, rB, dz_max=0, da_max=0, savedNodes=[], savedPanels=[]):
        '''Creates mesh for an axisymmetric member as defined by RAFT.

        Parameters
        ----------
        stations:  list 
            locations along member axis at which the cross section will be specified
        diameters: list 
            corresponding diameters along member
        rA, rB: list
            member end point coordinates
        dz_max: float
            maximum panel height
        da_max: float
            maximum panel width (before doubling azimuthal discretization)
        savedNodes : list of lists
            all the node coordinates already saved
        savedPanels : list
            the information for all the panels already saved: panel_number number_of_vertices   Vertex1_ID   Vertex2_ID  Vertex3_ID   (Vertex4_ID)


        Returns
        -------
        nodes : list
            list of node coordinates
        panels : list
            the information for all the panels: panel_number number_of_vertices   Vertex1_ID   Vertex2_ID  Vertex3_ID   (Vertex4_ID)

        '''

        def makePanel(X, Y, Z, savedNodes, savedPanels):
            '''
            Sets up panel and node data for HAMS .pnl input file.
            Also ensures things don't go above the water line. (A rough implementation that should be improved.)
        
            X, Y, Z : lists
                panel coordinates - 4 expected
            savedNodes : list of lists
                all the node coordinates already saved
            savedPanels : list
                the information for all the panels already saved: panel_number number_of_vertices   Vertex1_ID   Vertex2_ID  Vertex3_ID   (Vertex4_ID)
        
            '''
                    
            # now process the node points w.r.t. existing list
            #points = np.vstack([X, Y, Z])                # make a single 2D array for easy manipulation
            
            pNodeIDs = []                                    # the indices of the nodes for this panel (starts at 1)
            
            pNumNodes = 4
            
            for i in range(4):
            
                ndi = [X[i],Y[i],Z[i]]                       # the current node in question in the panel
                
                match = [j+1 for j,nd in enumerate(savedNodes) if nd==ndi]  # could do this in reverse order for speed...
                
                if len(match)==0:                            # point does not already exist in list; add it.
                    savedNodes.append(ndi)
                    pNodeIDs.append(len(savedNodes))
                    
                elif len(match)==1:                          # point exists in list; refer to it rather than adding another point
                    
                    if match[0] in pNodeIDs:                 # if the current panel has already referenced this node index, convert to tri
                        #print("triangular panel detected!")  # we will skip adding this point to the panel's indix list                                                     
                        pNumNodes -= 1                       # reduce the number of nodes for this panel
                        
                    else:
                        pNodeIDs.append(match[0])            # otherwise add this point index to the panel's list like usual
                    
                else:
                    ValueError("Somehow there are duplicate points in the list!")
                    
            panelID = len(savedPanels)+1                     # id number of the current panel (starts from 1)
            
            
            if pNumNodes == 4:
                savedPanels.append([panelID, pNumNodes, pNodeIDs[0], pNodeIDs[1], pNodeIDs[2], pNodeIDs[3]])
            elif pNumNodes == 3:
                savedPanels.append([panelID, pNumNodes, pNodeIDs[0], pNodeIDs[1], pNodeIDs[2]])
            else:
                ValueError(f"Somehow there are only {pNumNodes} unique nodes for panel {panelID}")

        radii = 0.5*np.array(diameters)


        # discretization defaults
        if dz_max==0:
            dz_max = stations[-1]/20
        if da_max==0:
            da_max = np.max(radii)/8


        # ------------------ discretize radius profile according to dz_max --------

        # radius profile data is contained in r_rp and z_rp
        r_rp = [radii[0]]
        z_rp = [0.0]

        # step through each station and subdivide as needed
        for i_s in range(1, len(radii)):
            dr_s = radii[i_s] - radii[i_s-1]; # delta r
            dz_s = stations[ i_s] - stations[ i_s-1]; # delta z
            # subdivision size
            if dr_s == 0: # vertical case
                cos_m=1
                sin_m=0
                dz_ps = dz_max; # (dz_ps is longitudinal dimension of panel)
            elif dz_s == 0: # horizontal case
                cos_m=0
                sin_m=np.sign(dr_s)
                dz_ps = 0.6*da_max
            else: # angled case - set panel size as weighted average based on slope
                m = dr_s/dz_s; # slope = dr/dz
                dz_ps = np.arctan(np.abs(m))*2/np.pi*0.6*da_max + np.arctan(abs(1/m))*2/np.pi*dz_max;
                cos_m = dz_s/np.sqrt(dr_s**2 + dz_s**2)
                sin_m = dr_s/np.sqrt(dr_s**2 + dz_s**2)
            # make subdivision
            # local panel longitudinal discretization
            n_z = int(np.ceil( np.sqrt(dr_s*dr_s + dz_s*dz_s) / dz_ps ))

            # local panel longitudinal dimension
            d_l = np.sqrt(dr_s*dr_s + dz_s*dz_s)/n_z
            for i_z in range(1,n_z+1):
                r_rp.append(  radii[i_s-1] + sin_m*i_z*d_l)
                z_rp.append(stations[i_s-1] + cos_m*i_z*d_l)


        # fill in end B if it's submerged
        n_r = int(np.ceil( radii[-1] / (0.6*da_max) ))   # local panel radial discretization #
        dr  = radii[-1] / n_r                               # local panel radial size

        for i_r in range(n_r):
            r_rp.append(radii[-1] - (1+i_r)*dr)
            z_rp.append(stations[-1])


        # fill in end A if it's submerged
        n_r = int(np.ceil( radii[0] / (0.6*da_max) ))   # local panel radial discretization #
        dr  = radii[0] / n_r                               # local panel radial size

        for i_r in range(n_r):
            r_rp.insert(0, radii[0] - (1+i_r)*dr)
            z_rp.insert(0, stations[0])


        # --------------- revolve radius profile, do adaptive paneling stuff ------

        # lists that we'll put the panel coordinates in, in lists of 4 coordinates per panel
        x = []
        y = []
        z = []

        npan =0
        naz = int(8)

        # go through each point of the radius profile, panelizing from top to bottom:
        for i_rp in range(len(z_rp)-1):
        
            # rectangle coords - shape from outside is:  A D
            #                                            B C
            r1=r_rp[i_rp]
            r2=r_rp[i_rp+1]
            z1=z_rp[i_rp]
            z2=z_rp[i_rp+1]

            # scale up or down azimuthal discretization as needed
            while ( (r1*2*np.pi/naz >= da_max/2) and (r2*2*np.pi/naz >= da_max/2) ):
                naz = int(2*naz)
            while ( (r1*2*np.pi/naz < da_max/2) and (r2*2*np.pi/naz < da_max/2) ):
                naz = int(naz/2)

            # transition - increase azimuthal discretization
            if ( (r1*2*np.pi/naz < da_max/2) and (r2*2*np.pi/naz >= da_max/2) ):
                for ia in range(1, int(naz/2)+1):
                    th1 = (ia-1  )*2*np.pi/naz*2
                    th2 = (ia-0.5)*2*np.pi/naz*2
                    th3 = (ia    )*2*np.pi/naz*2

                    x.append([(r1*np.cos(th1)+r1*np.cos(th3))/2, r2*np.cos(th2), r2*np.cos(th1), r1*np.cos(th1)])
                    y.append([(r1*np.sin(th1)+r1*np.sin(th3))/2, r2*np.sin(th2), r2*np.sin(th1), r1*np.sin(th1)])
                    z.append([z1                               , z2            , z2            , z1            ])

                    npan += 1

                    x.append([r1*np.cos(th3), r2*np.cos(th3), r2*np.cos(th2), (r1*np.cos(th1)+r1*np.cos(th3))/2])
                    y.append([r1*np.sin(th3), r2*np.sin(th3), r2*np.sin(th2), (r1*np.sin(th1)+r1*np.sin(th3))/2])
                    z.append([z1            , z2            , z2            , z1                               ])

                    npan += 1

            # transition - decrease azimuthal discretization
            elif ( (r1*2*np.pi/naz >= da_max/2) and (r2*2*np.pi/naz < da_max/2) ):
                for ia in range(1, int(naz/2)+1):
                    th1 = (ia-1  )*2*np.pi/naz*2
                    th2 = (ia-0.5)*2*np.pi/naz*2
                    th3 = (ia    )*2*np.pi/naz*2

                    x.append([r1*np.cos(th2), r2*(np.cos(th1)+np.cos(th3))/2, r2*np.cos(th1), r1*np.cos(th1)])
                    y.append([r1*np.sin(th2), r2*(np.sin(th1)+np.sin(th3))/2, r2*np.sin(th1), r1*np.sin(th1)])
                    z.append([z1            , z2                            , z2            , z1            ])

                    npan += 1

                    x.append([r1*np.cos(th3), r2*np.cos(th3), r2*(np.cos(th1)+np.cos(th3))/2, r1*np.cos(th2)])
                    y.append([r1*np.sin(th3), r2*np.sin(th3), r2*(np.sin(th1)+np.sin(th3))/2, r1*np.sin(th2)])
                    z.append([z1            , z2            , z2                            , z1            ])

                    npan += 1

            # no transition
            else:
                for ia in range(1, naz+1):
                    th1 = (ia-1)*2*np.pi/naz
                    th2 = (ia  )*2*np.pi/naz

                    x.append([r1*np.cos(th2), r2*np.cos(th2), r2*np.cos(th1), r1*np.cos(th1)])
                    y.append([r1*np.sin(th2), r2*np.sin(th2), r2*np.sin(th1), r1*np.sin(th1)])
                    z.append([z1            , z2            , z2            , z1            ])

                    npan += 1



        # calculate member rotation matrix 
        rAB = rB - rA                                               # displacement vector from end A to end B [m]

        beta = np.arctan2(rAB[1],rAB[0])                            # member incline heading from x axis
        phi  = np.arctan2(np.sqrt(rAB[0]**2 + rAB[1]**2), rAB[2])   # member incline angle from vertical

        s1 = np.sin(beta) 
        c1 = np.cos(beta)
        s2 = np.sin(phi) 
        c2 = np.cos(phi)
        s3 = np.sin(0.0) 
        c3 = np.cos(0.0)

        R = np.array([[ c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3,  c1*s2],
                      [ c1*s3+c2*c3*s1,  c1*c3-c2*s1*s3,  s1*s2],
                      [   -c3*s2      ,      s2*s3     ,    c2 ]])  #Z1Y2Z3 from https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

        nSavedPanelsOld = len(savedPanels)

        # adjust each panel position based on member pose then set up the member
        for i in range(npan):    
        
            nodes0 = np.array([x[i], y[i], z[i]])           # the nodes of this panel

            nodes = np.matmul( R, nodes0 ) + rA[:,None]    # rotate and translate nodes of this panel based on member pose

            makePanel(nodes[0], nodes[1], nodes[2], savedNodes, savedPanels)

        # print(f'Of {npan} generated panels, {len(savedPanels)-nSavedPanelsOld} were submerged and have been used in the mesh.')

        vertices = savedNodes
        faces = []

        for i in savedPanels:
            if i[1] == 3:
                faces.append([i[2], i[3], i[4], i[2]])
            elif i[1] == 4:
                faces.append([i[2], i[3], i[4], i[5]])
        
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=int) - 1
        
        return vertices, faces
    
    def show(self):
        '''Show mesh for the member.
        PARAMETERS
        ----------

        filename : str
            filename to be provided, currently VTP PNL and NEMOH.dat mesh is implemented
        '''
        from meshmagick.MMviewer import MMViewer
        from meshmagick.mmio import _build_vtkPolyData
        
        viewer = MMViewer()

        polydata = _build_vtkPolyData(self._vertices, self._faces)

        viewer.add_polydata(polydata)
        viewer.plane_on()
        viewer.show()

    def save(self, filename="MemberMesh.vtp"):
        '''Save mesh for the member.
        PARAMETERS
        ----------

        filename : str
            filename to be provided, currently VTP PNL and NEMOH.dat mesh is implemented
        '''
        if filename.endswith('.vtp'):
            from meshmagick.mmio import write_VTP
            write_VTP(filename, self._vertices, self._faces)

        elif filename.endswith('.pnl'):
            from meshmagick.mmio import write_PNL
            write_PNL(filename, self._vertices, self._faces)

        elif filename.endswith('.dat'):
            from meshmagick.mmio import write_MAR
            write_MAR(filename, self._vertices, self._faces)
        else:
            raise NotImplementedError("writter but 'VTP', 'PNL', 'NEMOH.dat' is not implemented yet")

    def getHydrostatics(self, rPRP=np.zeros(3), rho=1025, g=9.81):
        '''Get hydrostatics of a floating member.
        PARAMETERS
        ----------

        '''
        from meshmagick.mesh import Mesh
        from meshmagick.mesh_clipper import MeshClipper
        from meshmagick.hydrostatics import compute_hydrostatics

        def compute_hydrostatics_new(mesh, rPRP, rho_water, grav):

            wmesh = mesh.copy()

            # wcog = cog.copy()
            # wcog = np.dot(rotmat_corr, wcog)
            # wcog[2] += z_corr

            try:
                clipper = MeshClipper(wmesh, assert_closed_boundaries=True, verbose=False)
            except RuntimeError as err:
                raise err

            cmesh = clipper.clipped_mesh

            wet_surface_area = cmesh.faces_areas.sum()

            inertia = cmesh.eval_plain_mesh_inertias(rho_water)

            xb, yb, zb = inertia.gravity_center
            buoyancy_center = np.array([xb, yb, zb])
            disp_volume = inertia.mass / rho_water

            # Computing quantities from intersection polygons
            sigma0 = 0.  # \iint_{waterplane_area} dS = waterplane_area
            sigma1 = 0.  # \iint_{waterplane_area} x dS
            sigma2 = 0.  # \iint_{waterplane_area} y dS
            sigma3 = 0.  # \iint_{waterplane_area} xy dS
            sigma4 = 0.  # \iint_{waterplane_area} x^2 dS
            sigma5 = 0.  # \iint_{waterplane_area} y^2 dS
    
            xmin = []
            xmax = []
            ymin = []
            ymax = []
    
            polygons = clipper.closed_polygons
            for polygon in polygons:
                polyverts = clipper.clipped_crown_mesh.vertices[polygon]
    
            # To get the stiffness expressed at cog
            polyverts[:, 0] -= rPRP[0]
            polyverts[:, 1] -= rPRP[1]
    
            # TODO: voir si on conserve ce test...
            if np.any(np.fabs(polyverts[:, 2]) > 1e-3):
                print('The intersection polygon is not on the plane z=0')
    
            xi, yi = polyverts[0, :2]
            for (xii, yii) in polyverts[1:, :2]:
                dx = xii - xi
                dy = yii - yi
                px = xi + xii
                py = yi + yii
    
                sigma0 += dy * px
                sigma1 += dy * (px * px - xi * xii)
                sigma2 += dx * (py * py - yi * yii)
                sigma3 += dy * (py * px * px + yi * xi * xi + yii * xii * xii)
                sigma4 += dy * (xi * xi + xii * xii) * px
                sigma5 += dx * (yi * yi + yii * yii) * py
    
                xi, yi = xii, yii
    
            sigma0 /= 2
            sigma1 /= 6
            sigma2 /= -6
            sigma3 /= 24
            sigma4 /= 12
            sigma5 /= -12
    
            # Flotation surface
            waterplane_area = sigma0
    
            # Stiffness matrix coefficients that do not depend on the position of the gravity center
            rhog = rho_water * grav
            s33 = rhog * waterplane_area
            s34 = rhog * sigma2
            s35 = -rhog * sigma1
            s45 = -rhog * sigma3
    
            # Metacentric radius (Bouguer formulae)
            transversal_metacentric_radius = sigma5 / disp_volume  # Around Ox
            longitudinal_metacentric_radius = sigma4 / disp_volume  # Around Oy
    
            # Metacentric height
            zcog = rPRP[2]
            a = zcog - zb  # BG
            transversal_metacentric_height = transversal_metacentric_radius - a
            longitudinal_metacentric_height = longitudinal_metacentric_radius - a
    
            # Stiffness matrix coefficients that depend on the position of the gravity center
            s44 = rhog * disp_volume * transversal_metacentric_height
            s55 = rhog * disp_volume * longitudinal_metacentric_height

            # Assembling stiffness matrix
            stiffness_matrix = np.array([[ s33, -s34, -s35],
                                         [-s34,  s44, -s45],
                                         [-s35, -s45,  s55]], dtype=float)

            # Flotation center F:
            x_f = -s35 / s33 + rPRP[0]
            y_f =  s34 / s33 + rPRP[1]
    
            waterplane_center = np.array([x_f, y_f, 0])

            hs_data = {}
            hs_data['disp_mass'] = rho_water * disp_volume
            hs_data['buoyancy_center'] = buoyancy_center
            hs_data['stiffness_matrix'] = stiffness_matrix
            hs_data['disp_volume'] = disp_volume
            hs_data['waterplane_area'] = waterplane_area
            hs_data['waterplane_center'] = waterplane_center
            hs_data['transversal_metacentric_radius'] = transversal_metacentric_radius
            hs_data['longitudinal_metacentric_radius'] = longitudinal_metacentric_radius

            return hs_data


        mesh = Mesh(self._vertices, self._faces)
        mesh.heal_mesh()

        hsData = compute_hydrostatics_new(mesh, rPRP, grav=g, rho_water=rho)

        Fvec = np.zeros(6)
        Cmat = np.zeros([6,6])

        Fvec[2] = hsData["disp_mass"] * g
        Fvec[3] =  Fvec[2] * (hsData["buoyancy_center"][1]-rPRP[1])
        Fvec[4] = -Fvec[2] * (hsData["buoyancy_center"][0]-rPRP[0])
        
        Cmat[2:5,2:5] = hsData["stiffness_matrix"]
        # Cmat[2,3] = hsData["stiffness_matrix"][0,1]
        # Cmat[2,4] = hsData["stiffness_matrix"][0,2]
        # Cmat[3,2] = hsData["stiffness_matrix"][1,0]
        # Cmat[3,3] = hsData["stiffness_matrix"][1,1]
        # Cmat[3,4] = hsData["stiffness_matrix"][1,2]
        # Cmat[4,2] = hsData["stiffness_matrix"][2,0]
        # Cmat[4,3] = hsData["stiffness_matrix"][2,1]
        # Cmat[4,4] = hsData["stiffness_matrix"][2,2]

        V_UW = hsData["disp_volume"]
        r_center = hsData["buoyancy_center"]
        AWP = hsData["waterplane_area"]
        xWP = hsData["waterplane_center"][0]
        yWP = hsData["waterplane_center"][1]

        IxWP = hsData["transversal_metacentric_radius"] * V_UW
        IyWP = hsData["longitudinal_metacentric_radius"] * V_UW

        return Fvec, Cmat, V_UW, r_center, AWP, IxWP, IyWP, xWP, yWP
        
    def getMass(self, rho_medium=1023):

        from meshmagick.mesh import Mesh
        cal_mesh = Mesh(self.getVertices(), self.getFaces())
        body = cal_mesh.eval_plain_mesh_inertias(rho_medium)

        return body.mass, body.gravity_center
    
    def getVertices(self):

        return self._vertices
    
    def getFaces(self):

        return self._faces
    
    @property
    def nb_vertices(self) -> int:
        """Get the number of vertices in the mesh."""
        return self._vertices.shape[0]
    
    @property
    def nb_faces(self) -> int:
        """Get the number of faces in the mesh."""
        return self._faces.shape[0]

    def copy(self):

        return copy.deepcopy(self)


class platformMesh:

    def __init__(self, members=list[Member], meshMode=1, sizeMin=0.1, sizeMax=1.0, constraint=0.25, recombined=True, structured=False, clip=True):
        '''Initialize a platformMesh object from raft_member.Member object.
        MemberMesh objects are used to initialize 2D meshed for each member.

        PARAMETERS
        ----------

        members : list[raft_member.Member]
            list of Member objects
        meshMode : int
            mesh mode to be specified, currenlt 0 and 1, default=1:Gmsh support mesh
        sizeMin : float, optional
            Golbal mesh control of minimum size
        sizeMax : float, optional
            Golbal mesh control of maximum size
        constraint : float, optional
            Golbal mesh constraint by the scale of member  
        '''
        vertices = None
        faces = None

        for mem in members:

            memMesh = MemberMesh(mem, meshMode, sizeMin, sizeMax, constraint, recombined, structured, clip=clip)

            if vertices is None:
                vertices =memMesh.getVertices()
                faces = memMesh.getFaces()
            else:
                n = vertices.shape[0]
                vertices = np.vstack([vertices, memMesh.getVertices()])
                faces = np.vstack([faces, memMesh.getFaces()+n])      

        self._vertices = vertices
        self._faces = faces

    def show(self):
        '''Show mesh for the member.

        PARAMETERS
        ----------

        '''
        from meshmagick.MMviewer import MMViewer
        from meshmagick.mmio import _build_vtkPolyData
        
        viewer = MMViewer()

        polydata = _build_vtkPolyData(self._vertices, self._faces)

        viewer.add_polydata(polydata)
        viewer.plane_on()
        viewer.show()

    def save(self, filename="platformMesh.vtp"):
        '''Save mesh for the member.
        PARAMETERS
        ----------

        filename : str
            filename to be provided, currently VTP PNL and NEMOH.dat mesh is implemented
        '''
        if filename.endswith('.vtp'):
            from meshmagick.mmio import write_VTP
            write_VTP(filename, self._vertices, self._faces)

        elif filename.endswith('.pnl'):
            from meshmagick.mmio import write_PNL
            write_PNL(filename, self._vertices, self._faces)

        elif filename.endswith('.dat'):
            from meshmagick.mmio import write_MAR
            write_MAR(filename, self._vertices, self._faces)
        else:
            raise NotImplementedError("writter but 'VTP', 'PNL', 'NEMOH.dat' is not implemented yet")

    def getHydrostatics(self, rPRP=np.zeros(3), rho=1025, g=9.81):
        '''Get hydrostatics of a flaoting platform.
        PARAMETERS
        ----------

        rPRP : float array
            Coordinates of the platform reference point
        rho : float
            density of water
        g : float
            garvity accerlation
        '''
        from meshmagick.mesh import Mesh
        from meshmagick.hydrostatics import compute_hydrostatics

        mesh = Mesh(self._vertices, self._faces)

        hsData = compute_hydrostatics(mesh, cog=rPRP, grav=g, rho_water=rho)

        Fvec = np.zeros(6)
        Cmat = np.zeros([6,6])

        Fvec[2] = hsData["disp_mass"] * g
        Fvec[3] =  Fvec[2] * (hsData["buoyancy_center"][1]-rPRP[1])
        Fvec[4] = -Fvec[2] * (hsData["buoyancy_center"][0]-rPRP[0])
        
        Cmat[2,2] = hsData["stiffness_matrix"][0,0]
        Cmat[2,3] = -hsData["stiffness_matrix"][0,1]
        Cmat[2,4] = -hsData["stiffness_matrix"][0,2]
        Cmat[3,2] = -hsData["stiffness_matrix"][1,0]
        Cmat[3,3] = hsData["stiffness_matrix"][1,1]
        Cmat[3,4] = -hsData["stiffness_matrix"][1,2]
        Cmat[4,2] = -hsData["stiffness_matrix"][2,0]
        Cmat[4,3] = -hsData["stiffness_matrix"][2,1]
        Cmat[4,4] = hsData["stiffness_matrix"][2,2]

        V_UW = hsData["disp_volume"]
        r_center = hsData["buoyancy_center"]
        AWP = hsData["wet_surface_area"]

        return Fvec, Cmat

    def getVertices(self):

        return self._vertices

    def getFaces(self):

        return self._faces
    
    @property
    def nb_vertices(self) -> int:
        """Get the number of vertices in the mesh."""
        return self._vertices.shape[0]
    
    @property
    def nb_faces(self) -> int:
        """Get the number of faces in the mesh."""
        return self._faces.shape[0]

    def copy(self):

        return copy.deepcopy(self)
    
class sPlatformMesh(platformMesh): # single platform mesh, all members are attached together in Gmsh process

    def __init__ (self, mainMembers=list[Member], attchingMembers = list[Member], sizeMin=0.1, sizeMax=1.0, constraint=0.25, recombined=True, clip=True):
        import gmsh
        gmsh.initialize()

        mainList = []
        attachingList = []

        for index, mem in enumerate(mainMembers):
            mem = copy.deepcopy(mem)
            if mem.shape == "circular":
                self.__gmshCircMember(mem.stations, mem.d, mem.rA, mem.rB, index, sizeMin, sizeMax, constraint)
            elif mem.shape == "rectangular":
                self.__gmshRectMember(mem.stations, mem.sl, mem.rA, mem.R, index, sizeMin, sizeMax, constraint)

            mainList.append((3, index+1))

        for index_, mem_ in enumerate(attchingMembers):
            mem_ = copy.deepcopy(mem_)
            if mem_.shape == "circular":
                self.__gmshCircMember(mem_.stations, mem_.d, mem_.rA, mem_.rB, index_+index+1, sizeMin, sizeMax, constraint)
            elif mem_.shape == "rectangular":
                self.__gmshRectMember(mem_.stations, mem_.sl, mem_.rA, mem_.R, index_+index+1, sizeMin, sizeMax, constraint)

            attachingList.append((3, index_+index+2))

        if len(attachingList) > 1:
            gmsh.model.occ.fuse([attachingList[0]], attachingList[1:])
        gmsh.model.occ.synchronize()

        gmsh.model.occ.fuse([(3, len(mainList)+1)], mainList)
        gmsh.model.occ.synchronize()

        gmsh.fltk.run()

        # tag = gmsh.model.occ.getEntities(3)

        if clip == True:
            r = gmsh.model.occ.getBoundingBox(3,1)
            r0 = np.array([r[0],r[1]])
            r1 = np.array([r[3],r[4]])
            rc = (r1+r0) * 0.5
            rmax = np.linalg.norm(r1-rc)
            cutPlane = gmsh.model.occ.addCylinder(rc[0],rc[1], 0, 0, 0, 500, rmax*1.25)
            gmsh.model.occ.cut([(3,1)], [(3, cutPlane)])
            gmsh.model.occ.synchronize()
            # gmsh.fltk.run()

        gmsh.option.setNumber("Mesh.MeshSizeMin", sizeMin)
        gmsh.option.setNumber("Mesh.MeshSizeMax", sizeMax)
        # gmsh.model.mesh.setSize(gmsh.model.getEntities(0), np.min(sl)*constraint)
        gmsh.option.setNumber('Mesh.Format', 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.SaveParametric", 0)
        gmsh.option.setNumber("Mesh.AngleSmoothNormals", 8)

        if recombined:
            gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)
            gmsh.option.setNumber('Mesh.RecombineAll', 1)

        gmsh.model.mesh.generate(2)
        
        nodes, coords, _ = gmsh.model.mesh.getNodes()
        nodes, coords, _ = gmsh.model.mesh.getNodes()
        elementType, elements, elementNodes = gmsh.model.mesh.getElements()

        # gmsh.fltk.run()
        
        gmsh.finalize()

        nVertices = len(nodes)
        vertices = np.array(coords).reshape(nVertices, 3)

        if len(elementType[1:-1]) == 2: # triangular and quadature faces
            nTriFaces    = elements[1].shape[0]
            nQuadFaces   = elements[2].shape[0]

            faces_ = np.array(elementNodes[1]-1).reshape(nTriFaces ,3)
            triFaces = np.hstack((faces_, faces_[:, [0]]))

            quadFaces = np.array(elementNodes[2]-1).reshape(nQuadFaces ,4)

            faces = np.asarray(np.vstack((triFaces, quadFaces)))

        elif len(elementType[1:-1]) == 1 and elementType[1] == 2: # all triangular faces
            nFaces = elements[1].shape[0]

            faces_ = np.array(elementNodes[1]-1).reshape(nFaces ,3)
            faces = np.hstack((faces_, faces_[:, [0]]))

        elif len(elementType[1:-1]) == 1 and elementType[1] == 3: # all rectangular faces
            nFaces = elements[1].shape[0]

            faces = np.array(elementNodes[1]-1).reshape(nFaces ,4)

        self._vertices = vertices
        self._faces = faces



    def __gmshCircMember(self, stations, diameters, rA, rB, index, sizeMin=0.1, sizeMax=1, constraint=0.25, recombined=False, structured=False):
        '''Circular member meshing based on Gmsh.

        PARAMETERS
        ----------

        member : raft_member.Member
            A Member object initialized and position has been set
        nw : int
            Number of frequencies in the analysis - used for initializing.
        heading : float, optional
            Heading rotation to apply to the coordinates when setting up the 
            member. Used for member arrangements or FOWT heading offsets [deg].

        '''
        import gmsh

        q = (rB-rA)/stations[-1]
        radii = 0.5*np.array(diameters)
        rOrig = copy.deepcopy(rA)

        sections = [] # sections of discontinuous diameters

        for i in range(1, len(stations)):

            if stations[i] == stations[i-1]:
                pass

            else:

                dl = (stations[i]-stations[i-1]) * q   # the vector of ascending length
                dx,dy,dz = (dl[0], dl[1], dl[2])

                if radii[i] != radii[i-1]:
                    gmsh.model.occ.addCone(rOrig[0], rOrig[1],rOrig[2], dx, dy, dz, radii[i-1], radii[i], index+i)

                else:
                    gmsh.model.occ.addCylinder(rOrig[0], rOrig[1], rOrig[2], dx, dy, dz, radii[i-1], index+i)

                rOrig += q*(stations[i]-stations[i-1]) 
                sections.append((3,index+i))

        # gmsh.model.occ.synchronize()
        # gmsh.fltk.run()
        if len(sections) > 1:
            gmsh.model.occ.fuse([sections[0]], sections[1:])

        gmsh.model.occ.synchronize()
        # gmsh.fltk.run()
        # gmsh.option.setNumber("Mesh.MeshSizeMin", sizeMin)
        # gmsh.option.setNumber("Mesh.MeshSizeMax", sizeMax)
        # gmsh.model.mesh.setSize(gmsh.model.getEntities(0), np.min(diameters)*constraint)
        # gmsh.option.setNumber('Mesh.Format', 1)
        # gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        # gmsh.option.setNumber("Mesh.SaveParametric", 0)
        # gmsh.option.setNumber("Mesh.AngleSmoothNormals", 8)

        # if structured and recombined:
        #     gmsh.option.setNumber('Mesh.Algorithm', 11)
        # elif structured:
        #     gmsh.option.setNumber('Mesh.Algorithm', 11)
        # elif recombined:
        #     gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)
        #     gmsh.option.setNumber('Mesh.RecombineAll', 1)

    def __gmshRectMember(self, stations, sl, rA, R, index, sizeMin=0.1, sizeMax=1, constraint=0.25, recombined=False, structured=False):
        '''Rectangular member meshing based on Gmsh.

        PARAMETERS
        ----------

        member : raft_member.Member
            A Member object initialized and position has been set
        nw : int
            Number of frequencies in the analysis - used for initializing.
        heading : float, optional
            Heading rotation to apply to the coordinates when setting up the 
            member. Used for member arrangements or FOWT heading offsets [deg].

        '''
        import gmsh

        lx = 0.5*sl[:,0]
        ly = 0.5*sl[:,1]

        sections = [] # sections of discontinuous diameters 
        for i in range(1, len(stations)):

            rBot = np.array([[-lx[i-1], -ly[i-1], stations[i-1]],
                             [ lx[i-1], -ly[i-1], stations[i-1]],
                             [ lx[i-1],  ly[i-1], stations[i-1]],
                             [-lx[i-1],  ly[i-1], stations[i-1]]]).T
            rTop = np.array([[-lx[i],   -ly[i],   stations[i]],
                             [ lx[i],   -ly[i],   stations[i]],
                             [ lx[i],    ly[i],   stations[i]],
                             [-lx[i],    ly[i],   stations[i]]]).T
            
            rBot = np.matmul(R, rBot) + np.array([[rA[0]], [rA[1]], [rA[2]]])
            rTop = np.matmul(R, rTop) + np.array([[rA[0]], [rA[1]], [rA[2]]])

            if stations[i] == stations[i-1]:
                pass

            else:
                ptsBot = []
                ptsTop = []

                lineBot = []
                lineTop = []

                curvLoopList = []

                for k in range(4):
                    ptsBot.append(gmsh.model.occ.addPoint(rBot[0, k], rBot[1, k], rBot[2, k]))
                    ptsTop.append(gmsh.model.occ.addPoint(rTop[0, k], rTop[1, k], rTop[2, k]))

                for j in range(3):
                    lineBot.append(gmsh.model.occ.addLine(ptsBot[j], ptsBot[j+1]))
                    lineTop.append(gmsh.model.occ.addLine(ptsTop[j], ptsTop[j+1]))

                lineBot.append(gmsh.model.occ.addLine(ptsBot[-1], ptsBot[0]))
                lineTop.append(gmsh.model.occ.addLine(ptsTop[-1], ptsTop[0]))

                curvLoopList.append(gmsh.model.occ.addCurveLoop(lineBot))
                curvLoopList.append(gmsh.model.occ.addCurveLoop(lineTop))

                gmsh.model.occ.addThruSections(curvLoopList, index+i)

                sections.append((3,index+i))

        if len(sections) > 1:
            for sec in range (1, len(sections)):
                gmsh.model.occ.fuse([sections[0]], [sections[sec]])

        gmsh.model.occ.synchronize()



if __name__ == "__main__":

    list_files = [
    'mem_srf_vert_circ_cyl.yaml',    
    'mem_srf_vert_rect_cyl.yaml',
    'mem_srf_pitch_circ_cyl.yaml',
    'mem_srf_pitch_rect_cyl.yaml',
    'mem_srf_inc_circ_cyl.yaml',
    'mem_srf_inc_rect_cyl.yaml',
    'mem_subm_horz_circ_cyl.yaml',
    'mem_subm_horz_rect_cyl.yaml',
    'mem_srf_vert_tap_circ_cyl.yaml',
    'mem_srf_vert_tap_rect_cyl.yaml',
    ]

    # fname_design ="examples/OC4semi-RAFT_QTF.yaml"
    fname_design ="examples/VolturnUS-S_example.yaml"

    # fname_design = f"tests/test_data/{list_files[0]}"

    with open(fname_design) as file:
        design = yaml.load(file, Loader=yaml.FullLoader)

    # dict = design["members"][0]
    dict = design["platform"]["members"][1]
    # dict = design["platform"]["members"][2]
    mem_list = []
    mem_list.append(Member(dict, 1, heading=60))
    mem_list.append(Member(dict, 1, heading=180))
    mem_list.append(Member(dict, 1, heading=300))
    dict = design["platform"]["members"][0]
    mem_list.append(Member(dict, 1, heading=0))
    dict = design["platform"]["members"][2]
    mem_list.append(Member(dict, 1, heading=60))
    mem_list.append(Member(dict, 1, heading=180))
    mem_list.append(Member(dict, 1, heading=300))
    dict = design["platform"]["members"][3]
    mem_list.append(Member(dict, 1, heading=60))
    mem_list.append(Member(dict, 1, heading=180))
    mem_list.append(Member(dict, 1, heading=300))
    
    for mem in mem_list:
        mem.setPosition()
    # mem = Member(dict, 1)
    # mem.setPosition()
    # mesh = MemberMesh(mem, constraint=0.5)
    # mesh.show()

    from meshmagick.mesh import Mesh, Plane
    from meshmagick.mesh_clipper import MeshClipper
    from meshmagick.mmio import write_NEM, write_MAR, write_PNL

    mesh = platformMesh(mem_list[:7], sizeMax=3, constraint=0.5, clip=True)
    mesh.show()
    mesh = sPlatformMesh(mem_list[:4], mem_list[4:7], sizeMax=3, constraint=0.33)
    mesh.show()
    
    mesh = Mesh(mesh.getVertices(), mesh.getFaces())
    mesh = MeshClipper(mesh)
    # mesh = Mesh(mesh.clipped_mesh.vertices, mesh.clipped_mesh.faces)
    # mesh = MeshClipper(mesh, Plane([0,1,0]))

    mesh.clipped_mesh.show()
    write_PNL("Volturn.pnl", mesh.clipped_mesh.vertices, mesh.clipped_mesh.faces)
    write_MAR("Volturn.dat", mesh.clipped_mesh.vertices, mesh.clipped_mesh.faces)
    print(len(mesh.clipped_mesh.faces))

    write_NEM("Volturn", mesh.clipped_mesh.vertices, mesh.clipped_mesh.faces)



    from meshmagick.mmio import write_NEM


    mesh.show()
    mesh.getHydrostatics()

    mesh.save("0.dat")

    # mem_list = []
    # mem_list.append(Member(dict, 1).setPosition())
    # platform = platformMesh(mem_list, constraint=0.33, sizeMax=1.2, meshMode=1)
    # platform.show()
    # platform.save("5.dat")
    # Fvec, Cmat = platform.getHydrostatics(rPRP=[0,0,0])

    # print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in Cmat]))
    # result1 = mem.getHydrostatics()
    # print(f"waterplane area = {result1[4]} \n")
    # print(f"C44 = {result1[1][3,3]} \n")
    # print(f"C55 = {result1[1][4,4]} \n")

    # result2 = mem.getHydrostaticsYang()
    # print(f"waterplane area = {result2[4]} \n")
    # print(f"C44 = {result2[1][3,3]} \n")
    # print(f"C55 = {result2[1][4,4]} \n")

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # # ax.axis('equal')
    # mem.plot(ax)
    # ax.axis('equal')
    
    # plt.xlabel("x")
    # plt.show()

    # mesh = MemberMesh(mem, 0)
    # mesh.show()

    # mesh = MemberMesh(mem, 1, constraint=0.33)
    # mesh.show()

    # Fvec, Cmat = mesh.getHydrostatics()

    # print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in Cmat]))

    # result1 = mem.getHydrostatics()
    # result1 = mem.getHydrostaticsYang()
    a = 1

    # print("###################################################")
    # print(mem.getMemberInertia())
    # a = 1

