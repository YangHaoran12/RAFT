"""BEM solvers for solving diffarction-radiation problems (temporary 1st order)."""
from ntpath import join
import numpy as np
import xarray as xr
from raft.raft_member import Member, TowerMember
from raft.raft_mesh import platformMesh, sPlatformMesh
import os

from typing import Iterable

try: 
    import capytaine as cpt
except:
        raise ModuleNotFoundError("module capytain can't be found")
from capytaine.bodies import FloatingBody
from capytaine.bem.problems_and_results import RadiationProblem

# global constants
raft_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
rad2deg = 57.2958

class BEMSolver(object):
    """General class of a BEM solver.

        At most one of the following parameter must be provided: omega, period, wavenumber or wavelength.
        Internally only omega is stored, hence setting another parameter can lead to small rounding errors.

        Parameters
        ----------
        body: FloatingBody, optional
            The body interacting with the waves
        free_surface: float, optional
            The position of the free surface (accepted values: 0 and np.inf)
        water_depth: float, optional
            The depth of water in m (default: np.inf)
        sea_bottom: float, optional
            The position of the sea bottom (deprecated: please prefer setting water_depth)
        omega: float, optional
            The angular frequency of the waves in rad/s
        period: float, optional
            The period of the waves in s
        wavenumber: float, optional
            The angular wave number of the waves in rad/m
        wavelength: float, optional
            The wave length of the waves in m
        forward_speed: float, optional
            The speed of the body (in m/s, in the x direction, default: 0.0)
        rho: float, optional
            The density of water in kg/m3 (default: 1000.0)
        g: float, optional
            The acceleration of gravity in m/s2 (default: 9.81)
        boundary_condition: np.ndarray of shape (body.mesh.nb_faces,), optional
            The Neumann boundary condition on the floating body
        """
    def __init__(self, 
                 memList:list[Member], 
                 wMin=0.0618, 
                 wMax=6.18, 
                 nw=10,
                 headings=[0], # deg
                 cog=[0,0,0],  # center of gravity
                 rho=1025,
                 g=9.81,
                 waterDepth=200,
                 include0andinf=True,
                 sizeMin=0.1, 
                 sizeMax=1.0, 
                 constraint=0.25, 
                 singleBody=False,
                 meshDir=os.path.join(os.getcwd(),'BEM'),
                 showMesh=False,
                 saveMesh=False):
        
        self.w = np.linspace(wMin, wMax, nw, endpoint=True) # rad/s
        self.headings = headings
        self.cog = cog
        self.rho = rho
        self.g = g
        self.waterDepth = waterDepth
        self.include0andInf = include0andinf
        
        self.A = None
        self.B = None
        self.M = None
        self.P = None
        self.R = None
        self.I = None
        

        if memList is not None:
            
            if singleBody is True: # a fuse-member solution under tests, not recommended!
                self.mesh = sPlatformMesh(mainMembers=memList[:4],
                                          attchingMembers=memList[4:],
                                          sizeMin=sizeMin, 
                                          sizeMax=sizeMax,
                                          constraint=constraint,
                                          recombined=True,
                                          clip=True
                                          )
            else:
                self.mesh = platformMesh(members=memList,
                                         sizeMin=sizeMin, 
                                         sizeMax=sizeMax,
                                         constraint=constraint,
                                         recombined=True,
                                         clip=True
                                        )
        else:

            self.mesh = None
        
        if showMesh is True:
            self.mesh.show()

        if saveMesh is True:
            self.mesh.save(os.path.join(meshDir, 'platform.dat'))

    
    def plotRadiationResults(self, fType='omega'):
        '''
        solve capytaine problems in thread parallel

        mesh format: .dat(Nemoh) and .msh(gmsh)

        Parameters
        ----------
        meshFile: str, os.path
            the mesh file
        '''
        A11 = self.A[:,0,0]
        A22 = self.A[:,1,1]
        A33 = self.A[:,2,2]
        A44 = self.A[:,3,3]
        A55 = self.A[:,4,4]
        A66 = self.A[:,5,5]

        A15 = self.A[:,0,4]
        A24 = self.A[:,1,3]

        B11 = self.B[:,0,0]
        B22 = self.B[:,1,1]
        B33 = self.B[:,2,2]
        B44 = self.B[:,3,3]
        B55 = self.B[:,4,4]
        B66 = self.B[:,5,5]

        B15 = self.B[:,0,4]
        B24 = self.B[:,1,3]

        import matplotlib.pyplot as plt

        if fType == 'omega':
            f = self.w
        elif fType == 'frequency':
            f = self.w/2/np.pi
        else:
            raise NotImplementedError('currently only omega[rad] and frequency[Hz] are implemented')
        
        if self.include0andInf:

            f = np.concatenate(([0.0], f, [self.w[-1]+0.1]))

        fig,ax = plt.subplots(3,2)
        ax[0,0].plot(f, A11, color='black')
        ax[0,0].plot(f, A22, color='red'  )
        ax[0,0].plot(f, A33, color='blue' )
        ax[0,0].set_xticklabels([])
        ax[0,0].grid(True)
        ax[0,0].legend(['A11', 'A22', 'A33'])

        ax[0,1].plot(f, B11, color='black')
        ax[0,1].plot(f, B22, color='red'  )
        ax[0,1].plot(f, B33, color='blue' )
        ax[0,1].set_xticklabels([])
        ax[0,1].grid(True)
        ax[0,1].legend(['B11', 'B22', 'B33'])

        ax[1,0].plot(f, A44, color='black')
        ax[1,0].plot(f, A55, color='red'  )
        ax[1,0].plot(f, A66, color='blue' )
        ax[1,0].set_xticklabels([])
        ax[1,0].grid(True)
        ax[1,0].legend(['A44', 'A55', 'A66'])

        ax[1,1].plot(f, B44, color='black')
        ax[1,1].plot(f, B55, color='red'  )
        ax[1,1].plot(f, B66, color='blue' )
        ax[1,1].set_xticklabels([])
        ax[1,1].grid(True)
        ax[1,1].legend(['B44', 'B55', 'B66'])

        ax[2,0].plot(f, A15, color='black')
        ax[2,0].plot(f, A24, color='red'  )
        ax[2,0].grid(True)
        ax[2,0].legend(['A24', 'A15'])

        ax[2,1].plot(f, B15, color='black')
        ax[2,1].plot(f, B24, color='red'  )
        ax[2,1].grid(True)
        ax[2,1].legend(['B24', 'B15'])

        ax[2,0].set_xlabel('Frequency [rad/s]')
        ax[2,1].set_xlabel('Frequency [rad/s]')

        plt.show()

        a = 1


    def plotFexcResults(self, fType='omega', nHeading=0):
        '''
        solve capytaine problems in thread parallel

        mesh format: .dat(Nemoh) and .msh(gmsh)

        Parameters
        ----------
        meshFile: str, os.path
            the mesh file
        '''

        F = self.M[:, nHeading, :]
        # P = np.angle(self.R[:, nHeading, :] + self.I[:, nHeading, :]*1j, deg=True)
        P = self.P[:, nHeading, :]
        
        F1 = F[:,0]
        F2 = F[:,1]
        F3 = F[:,2]
        M1 = F[:,3]
        M2 = F[:,4]
        M3 = F[:,5]

        PF1 = P[:,0]
        PF2 = P[:,1]
        PF3 = P[:,2]
        PM1 = P[:,3]
        PM2 = P[:,4]
        PM3 = P[:,5]

        if fType == 'omega':
            f = self.w
        elif fType == 'frequency':
            f = self.w/2/np.pi
        else:
            raise NotImplementedError('currently only omega[rad] and frequency[Hz] are implemented')

        import matplotlib.pyplot as plt

        fig,ax = plt.subplots(2,2)

        ax[0,0].plot(f, F1, color='black')
        ax[0,0].plot(f, F2, color='red'  )
        ax[0,0].plot(f, F3, color='blue' )
        ax[0,0].set_xlabel('Frequency [rad/s]')
        ax[0,0].grid(True)
        ax[0,0].legend(['Surge', 'Sway', 'Heave'])
        
        ax[0,1].plot(f, M1, color='black')
        ax[0,1].plot(f, M2, color='red'  )
        ax[0,1].plot(f, M3, color='blue' )
        ax[0,1].set_xlabel('Frequency [rad/s]')
        ax[0,1].grid(True)
        ax[0,1].legend(['Roll', 'Pitch', 'Yaw'])

        ax[1,0].plot(f, PF1, color='black')
        ax[1,0].plot(f, PF2, color='red'  )
        ax[1,0].plot(f, PF3, color='blue' )
        ax[1,0].legend(['Surge', 'Sway', 'Heave'])
        
        ax[1,1].plot(f, PM1, color='black')
        ax[1,1].plot(f, PM2, color='red'  )
        ax[1,1].plot(f, PM3, color='blue' )
        ax[1,1].legend(['Roll', 'Pitch', 'Yaw'])

        plt.show()

    def getResultsFromWamit(self, WamitDir:str, FileName:str):

        Wamit1File = os.path.join(WamitDir, FileName + '.1')
        Wamit3File = os.path.join(WamitDir, FileName + '.3')

        Acoef, Bcoef, w1 = readWamit1(Wamit1File)

        if any(w1 == 0.0) and any(w1 == np.inf):
            w1 = np.concatenate(([0.], w1[1:-1], [w1[-2]+0.1]))
        
        self.A = self.rho*Acoef
        self.B = self.rho*w1[:, np.newaxis, np.newaxis]*Bcoef

        mod, phase, real, imag, w3, headings = readWamit3(Wamit3File)

        self.M = mod*self.rho*self.g
        self.P = phase
        self.R = real*self.rho*self.g
        self.I = imag*self.rho*self.g

        self.w = w3
        self.headings = headings

class CapytaineSolver(BEMSolver):

    def __init__(self, 
                 memList: Iterable[Member], 
                 wMin=0.0618, 
                 wMax=6.18, 
                 nw=10, 
                 headings=[0], 
                 cog=[0, 0, 0], 
                 rho=1025, 
                 g=9.81, 
                 waterDepth=200, 
                 include0andinf=True, 
                 sizeMin=0.1, 
                 sizeMax=1, 
                 constraint=0.25, 
                 singleBody=False, 
                 meshDir=os.path.join(os.getcwd(), 'BEM'), 
                 showMesh=False, 
                 saveMesh=False):
        
        super().__init__(memList, wMin, wMax, nw, headings, cog, rho, g, waterDepth, include0andinf, sizeMin, sizeMax, constraint, singleBody, meshDir, showMesh, saveMesh)

        if self.mesh is not None:

            self.getCapytaineMeshFromMesh()

    def getCapytaineMeshFromMesh(self):
        '''
        initialize a cal_mesh and lid_mesh from self.mesh

        mesh format: .dat(Nemoh) and .msh(gmsh)

        Parameters
        ----------
        meshFile: str, os.path
            the mesh file
        '''
        hull_mesh = cpt.Mesh(self.mesh.getVertices(), self.mesh.getFaces())
        cal_mesh, lid_mesh = hull_mesh.extract_lid()

        if lid_mesh.nb_faces == 0:
            print('Warning: lid mesh has not been generated properly, please check the orinial hull_mesh!')

        lid_mesh.flip_normals() # flip normal to avoid capytaine body warning, not neccessary

        self.cal_mesh = cal_mesh
        self.lid_mesh = lid_mesh

    def getCapytaineMeshFromFile(self, meshFile):
        '''
        initialize a cal_mesh and lid_mesh from meshFile

        mesh format: .dat(Nemoh) and .msh(gmsh)

        Parameters
        ----------
        meshFile: str, os.path
            the mesh file
        '''

        hull_mesh = cpt.load_mesh(meshFile)
        hull_mesh.heal_mesh()
        
        cal_mesh, lid_mesh = hull_mesh.extract_lid()

        if lid_mesh.nb_faces == 0:
            print('Warning: lid mesh has not been generated properly, please check the orinial hull_mesh!')

        lid_mesh.flip_normals()

        self.cal_mesh = cal_mesh
        self.lid_mesh = lid_mesh

    def defineProblems(self):
        '''
        initialize capytaine problems

        mesh format: .dat(Nemoh) and .msh(gmsh)

        Parameters
        ----------
        meshFile: str, os.path
            the mesh file
        '''
        problems = []

        if self.include0andInf:
            # initialize a body with out lid, since lid_mesh is not supported in radiation problems

            body = cpt.FloatingBody(self.cal_mesh, dofs=cpt.rigid_body_dofs(rotation_center=self.cog), center_of_mass = self.cog, name='platform')
            
            problems0andInf = [cpt.RadiationProblem(omega=0.0, body=body, radiating_dof=dof, g=self.g, rho=self.rho) for dof in body.dofs]
            problems0andInf.extend([cpt.RadiationProblem(omega=np.inf, body=body, radiating_dof=dof, g=self.g, rho=self.rho) for dof in body.dofs])

            self.problems0andInf = problems0andInf
        else:

            self.problems0andInf = None

        
        body = cpt.FloatingBody(mesh=self.cal_mesh, 
                                lid_mesh=self.lid_mesh, 
                                dofs=cpt.rigid_body_dofs(rotation_center=self.cog), 
                                center_of_mass = self.cog, 
                                name="platform")
        
        test_matrix = xr.Dataset(coords={
                            'omega': self.w,
                            'rho':self.rho,
                            'wave_direction': np.deg2rad(self.headings), # deg to rad when define problems
                            'radiating_dof': list(body.dofs),
                            'water_depth': self.waterDepth,
                                })
        from capytaine.io.xarray import problems_from_dataset

        problems.extend(problems_from_dataset(test_matrix, body))

        self.problems = problems
    
    def solveProblems(self, nThreads):
        '''
        solve capytaine problems in thread parallel

        mesh format: .dat(Nemoh) and .msh(gmsh)

        Parameters
        ----------
        meshFile: str, os.path
            the mesh file
        '''
        from capytaine.bem.solver import BEMSolver
        from capytaine.io.xarray import assemble_dataset

        results = BEMSolver().solve_all(problems=self.problems, n_jobs=nThreads)
        dataset = assemble_dataset(results)

        if self.problems0andInf is not None:
            results0andInf = BEMSolver().solve_all(problems=self.problems0andInf, n_jobs=4)
            dataset0andInf = assemble_dataset(results0andInf)

            dataset0andInf = dataset0andInf.assign_coords(water_depth = self.waterDepth,  # assign water_depth for w = 0 and inf to self.waterDepth
                                                      wave_direction = self.headings) # assign wave_direction for w = 0 and inf to self.headings
                                                                                      # only for convenience
            self.dataset = xr.merge([dataset0andInf, dataset])
        else:
            self.dataset = dataset

    def postProcess(self):
        '''
        solve capytaine problems in thread parallel

        mesh format: .dat(Nemoh) and .msh(gmsh)

        Parameters
        ----------
        meshFile: str, os.path
            the mesh file
        '''

        A = self.dataset['added_mass'].data
        B = self.dataset['radiation_damping'].data

        if self.include0andInf:
            B[-1,:,:] = np.zeros((6,6)) # a bug in capytaine will return inf damping when w=inf
                                        # mannually solve it

            Fexc = self.dataset['excitation_force'].data[1:-1] # mannually remove nan in 0 and inf frequency 

        M = np.abs(Fexc)
        P = np.angle(np.conj(Fexc), deg=True)
        R = np.real(Fexc)
        I = np.imag(Fexc)

        self.A = A
        self.B = B
        self.M = M
        self.P = P
        self.R = R
        self.I = I

    def writeWamit(self, projectDir=None, WamitFile='Platform'):
        projectDir = os.path.join(raft_dir, projectDir)
        
        os.makedirs(projectDir, exist_ok=True)
        
        f = open(os.path.join(projectDir, WamitFile+'.1'), 'w')

        if self.include0andInf:
            for i in range(6):
                for j in range(6):

                    f.write(f'{0.0:>13.6E}    {i+1:>4}    {j+1:>4}    {self.A[0,i,j]/self.rho:>13.6E} \n')

            for i in range(6):
                for j in range(6):

                    f.write(f'{-1.0:>13.6E}    {i+1:>4}    {j+1:>4}    {self.A[-1,i,j]/self.rho:>13.6E} \n')


        for iw, w in enumerate(self.w):
            if self.include0andInf:
                iw += 1
            for i in range(6):
                for j in range(6):

                    f.write(f'{w:>13.6E}    {i+1:>4}    {j+1:>4}    {self.A[iw,i,j]/self.rho:>13.6E}    {self.B[iw,i,j]/self.rho/w:>13.6E} \n')
        f.close()

        f = open(os.path.join(projectDir, WamitFile+'.3'), 'w')

        for iw, w in enumerate(self.w):
            for ih, h in enumerate(self.headings):
                for i in range(6):
                    
                    f.write(f'{w:>13.6E}    '
                            f'{h:>13.6E}    '
                            f'{i+1:>4}    '
                            f'{self.M[iw,ih,i]/self.rho/self.g:>13.6E}    '
                            f'{self.P[iw,ih,i]:>13.6E}    ' # convert rad to degree using a less accurate method
                            f'{self.R[iw,ih,i]/self.rho/self.g:>13.6E}    '
                            f'{self.I[iw,ih,i]/self.rho/self.g:>13.6E} \n')
        f.close()

    def plotRadiationResults(self, fType='omega'):
        '''
        solve capytaine problems in thread parallel

        mesh format: .dat(Nemoh) and .msh(gmsh)

        Parameters
        ----------
        meshFile: str, os.path
            the mesh file
        '''
        A11 = self.dataset['added_mass'].sel(radiating_dof='Surge', influenced_dof='Surge')
        A22 = self.dataset['added_mass'].sel(radiating_dof='Sway', influenced_dof='Sway')
        A33 = self.dataset['added_mass'].sel(radiating_dof='Heave', influenced_dof='Heave')
        A44 = self.dataset['added_mass'].sel(radiating_dof='Roll', influenced_dof='Roll')
        A55 = self.dataset['added_mass'].sel(radiating_dof='Pitch', influenced_dof='Pitch')
        A66 = self.dataset['added_mass'].sel(radiating_dof='Yaw', influenced_dof='Yaw')

        A15 = self.dataset['added_mass'].sel(radiating_dof='Surge', influenced_dof='Pitch')
        A24 = self.dataset['added_mass'].sel(radiating_dof='Sway', influenced_dof='Roll')

        B11 = self.dataset['radiation_damping'].sel(radiating_dof='Surge', influenced_dof='Surge')
        B22 = self.dataset['radiation_damping'].sel(radiating_dof='Sway', influenced_dof='Sway')
        B33 = self.dataset['radiation_damping'].sel(radiating_dof='Heave', influenced_dof='Heave')
        B44 = self.dataset['radiation_damping'].sel(radiating_dof='Roll', influenced_dof='Roll')
        B55 = self.dataset['radiation_damping'].sel(radiating_dof='Pitch', influenced_dof='Pitch')
        B66 = self.dataset['radiation_damping'].sel(radiating_dof='Yaw', influenced_dof='Yaw')

        B15 = self.dataset['radiation_damping'].sel(radiating_dof='Surge', influenced_dof='Pitch')
        B24 = self.dataset['radiation_damping'].sel(radiating_dof='Sway', influenced_dof='Roll')

        import matplotlib.pyplot as plt

        if fType == 'omega':
            f = self.w
        elif fType == 'frequency':
            f = self.w/2/np.pi
        else:
            raise NotImplementedError('currently only omega[rad] and frequency[Hz] are implemented')
        
        if self.include0andInf:

            f = np.concatenate(([1e-3], f, [self.w[-1]+0.1]))

        fig,ax = plt.subplots(3,2)
        ax[0,0].plot(f, A11)
        ax[0,0].plot(f, A22)
        ax[0,0].plot(f, A33)
        ax[0,0].legend(['A11', 'A22', 'A33'])

        ax[0,1].plot(f, B11)
        ax[0,1].plot(f, B22)
        ax[0,1].plot(f, B33)
        ax[0,1].legend(['B11', 'B22', 'B33'])

        ax[1,0].plot(f, A44)
        ax[1,0].plot(f, A55)
        ax[1,0].plot(f, A66)
        ax[1,0].legend(['A44', 'A55', 'A66'])

        ax[1,1].plot(f, B44)
        ax[1,1].plot(f, B55)
        ax[1,1].plot(f, B66)
        ax[1,1].legend(['B44', 'B55', 'B66'])

        ax[2,0].plot(f, A24)
        ax[2,0].plot(f, A15)
        ax[2,0].legend(['A24', 'A15'])

        ax[2,1].plot(f, B24)
        ax[2,1].plot(f, B15)
        ax[2,1].legend(['B24', 'B15'])

        plt.show()


    def plotFexcResults(self, fType='omega', nHeading=0):
        '''
        solve capytaine problems in thread parallel

        mesh format: .dat(Nemoh) and .msh(gmsh)

        Parameters
        ----------
        meshFile: str, os.path
            the mesh file
        '''
        if self.include0andInf:
            Fexc = self.dataset['excitation_force'].isel(wave_direction=nHeading).isel(omega=range(1, len(self.w)+1))
        else:
            Fexc = self.dataset['excitation_force'].isel(wave_direction=nHeading)
        
        F1 = Fexc.sel(influenced_dof='Surge')
        F2 = Fexc.sel(influenced_dof='Sway')
        F3 = Fexc.sel(influenced_dof='Heave')
        M1 = Fexc.sel(influenced_dof='Roll')
        M2 = Fexc.sel(influenced_dof='Pitch')
        M3 = Fexc.sel(influenced_dof='Yaw')

        if fType == 'omega':
            f = self.w
        elif fType == 'frequency':
            f = self.w/2/np.pi
        else:
            raise NotImplementedError('currently only omega[rad] and frequency[Hz] are implemented')

        import matplotlib.pyplot as plt

        fig,ax = plt.subplots(2,2)

        ax[0,0].plot(f, np.abs(F1))
        ax[0,0].plot(f, np.abs(F2))
        ax[0,0].plot(f, np.abs(F3))
        ax[0,0].legend(['Surge', 'Sway', 'Heave'])
        
        ax[0,1].plot(f, np.abs(M1))
        ax[0,1].plot(f, np.abs(M2))
        ax[0,1].plot(f, np.abs(M3))
        ax[0,1].legend(['Roll', 'Pitch', 'Yaw'])

        ax[1,0].plot(f, np.angle(F1, deg=True))
        ax[1,0].plot(f, np.angle(F2, deg=True))
        ax[1,0].plot(f, np.angle(F3, deg=True))
        ax[1,0].legend(['Surge', 'Sway', 'Heave'])
        
        ax[1,1].plot(f, np.angle(M1, deg=True))
        ax[1,1].plot(f, np.angle(M2, deg=True))
        ax[1,1].plot(f, np.angle(M3, deg=True))
        ax[1,1].legend(['Roll', 'Pitch', 'Yaw'])

        plt.show()



##################################################################################
def readWamit1(Wamit1File):
    '''
    Read added mass and damping from .1 file (WAMIT format)

    Parameters
    ----------
    Wamit1File: str, os.path
        Wamit .1 file
    '''
    Wamit1File = os.path.normpath(os.path.join(raft_dir, Wamit1File))
    # Check if the file contains the infinite and zero frequency points
    try:
        # extract the first 72 rows to see how many are infinite and zero
        freq_test = np.loadtxt(Wamit1File, usecols=(0,1,2,3), max_rows=73)  # add one extra (73) for formatting/syntax
        # find the row in the file that is not a zero or infinite frequency
        for i in range(len(freq_test)):     
            if freq_test[i,0] != 0.0 and freq_test[i,0] != -1.0:
                break
        # create new arrays based on the number of nonzero values in the matrices
        inf_zero_freqs = np.loadtxt(Wamit1File, usecols=(0,1,2,3), max_rows=i)
        other_freqs = np.loadtxt(Wamit1File, usecols=(0,1,2,3,4), skiprows=i)
        
        zero_freq   = np.c_[inf_zero_freqs[:36], np.zeros(36)]
        
        inf_freq   = np.c_[inf_zero_freqs[36:], np.zeros(36)] # force Raidation damping coefficient Bij^hat in 0 and inf frequncy to be 0, for interpolation convenience
        inf_freq[:, 0] = np.inf # set inf_freq to np.inf
        
        wamit1      = np.vstack((zero_freq, other_freqs, inf_freq))
    except:
        wamit1 = np.loadtxt(Wamit1File)
    # Get unique frequencies in a sorted order
    iw = np.argsort(np.unique(wamit1[:,0]))
    w = wamit1[:,0][::36][iw]
    nfreq = len(w)

    addedMassCol = wamit1[:,3]
    dampingCol   = wamit1[:,4]

    ACoef = addedMassCol.reshape(nfreq, 36).reshape((nfreq,6,6))[iw]
    BCoef = dampingCol.reshape(nfreq, 36).reshape((nfreq,6,6))[iw]

    return ACoef, BCoef, w

def readWamit3(Wamit3File):
    '''
    Read excitation force coefficients from .3 file (WAMIT format)

    Parameters
    ----------
    Wamit1File: str, os.path
        Wamit .3 file
    '''
    
    Wamit3File = os.path.normpath(os.path.join(raft_dir, Wamit3File))
    wamit3 = np.loadtxt(Wamit3File)

    # Get unique frequencies and index vector
    iw = np.argsort(np.unique(wamit3[:,0]))
    ih = np.argsort(np.unique(wamit3[:,1]))

    nfreq = len(iw)
    nheadings = len(ih)

    w = wamit3[:,0][::6*nheadings][iw]
    headings = wamit3[:,1][0:6*nheadings:6][ih]
        
    # headings, ih = np.unique(wamit3[:,1], return_inverse=True)
    # nhead = len(headings)

    modCol   = wamit3[:,3]
    phaseCol = wamit3[:,4]
    realCol  = wamit3[:,5]
    imagCol  = wamit3[:,6]
    
    mod   = modCol.reshape(nfreq, nheadings, 6)[iw][:,ih,:]
    phase = phaseCol.reshape(nfreq, nheadings, 6)[iw][:,ih,:]
    real  = realCol.reshape(nfreq, nheadings, 6)[iw][:,ih,:]
    imag  = imagCol.reshape(nfreq, nheadings, 6)[iw][:,ih,:]
    
    return mod, phase, real, imag, w, headings

def animation(calMeshFile=os.path.join(raft_dir, 'BEM/VolturnUS-S/platform.dat'), 
              platform=True, 
              platformMeshFile=os.path.join(raft_dir, 'models/nemoh/OC4_full.dat'), 
              tower=True,
              towerMeshFile=os.path.join(raft_dir, 'models/nemoh/OC4_tower.dat'), 
              rotor=True,
              rotorMeshFile=os.path.join(raft_dir, 'models/nemoh/NREL_5MW_rotor.dat'),
              hHub=90,
              omega:float = 1.0,
              heading:float = 0.,
              eta:float=1.0,
              rao=np.zeros(6),
              outFile=None
              ):
    
    from capytaine import load_mesh
    from capytaine import FreeSurface
    from capytaine.bem.airy_waves import airy_waves_free_surface_elevation
    from capytaine.ui.vtk import Animation

    if platform:

        platformMesh = load_mesh(platformMeshFile)
        platform_body = cpt.FloatingBody(mesh=platformMesh, dofs=cpt.rigid_body_dofs(rotation_center=[0,0,0]), center_of_mass=[0,0,0])
    else:

        platform_body = None

    if rotor:

        rotorMesh    = load_mesh(rotorMeshFile)
        rotor_body = cpt.FloatingBody(mesh=rotorMesh, dofs=cpt.rigid_body_dofs(rotation_center=[0,0,hHub]), center_of_mass=[0,0,0])

    else:
        
        rotor_body = None

    if tower:

        towerMesh = load_mesh(towerMeshFile)
        tower_body = cpt.FloatingBody(mesh=towerMesh, dofs=cpt.rigid_body_dofs(rotation_center=[0,0,0]), center_of_mass=[0,0,0])
       

    hull_mesh    = load_mesh(calMeshFile)

    ani_body = cpt.FloatingBody.join_bodies(platform_body, tower_body)
    ani_body.add_all_rigid_body_dofs()


    import copy
    from capytaine.meshes.geometry import Axis

    Animation.add_rotor = classmethod(add_rotor)

    cal_mesh, lid_mesh = hull_mesh.extract_lid()

    cal_body = cpt.FloatingBody(cal_mesh, dofs=cpt.rigid_body_dofs(rotation_center=[0,0,0]), center_of_mass=[0,0,0])

    # cal_body.inertia_matrix = cal_body.compute_rigid_body_inertia()*np.diag((1e-3,1e-3,1e-3,1e-7,1e-7,1e-7)) # Artificially lower to have a more appealing animation
    # cal_body.hydrostatic_stiffness = cal_body.compute_hydrostatic_stiffness()

    fs = FreeSurface(x_range=(-200.0, 200.0), nx=100, y_range=(-200.0, 200.0), ny=100)

    # animation = Animation(loop_duration=2*np.pi/omega)
    # animation.add_body(ani_body)
    # animation.run(camera_position=(-75*5/3,-75,350))

    solver = cpt.BEMSolver()

    radiation_problems = [cpt.RadiationProblem(omega=omega, body=cal_body, radiating_dof=dof) for dof in cal_body.dofs]
    radiation_results = solver.solve_all(radiation_problems)

    diffraction_problem = cpt.DiffractionProblem(omega=omega, body=cal_body, wave_direction=np.deg2rad(heading))
    diffraction_result = solver.solve(diffraction_problem)

    dataset = cpt.assemble_dataset(radiation_results + [diffraction_result])
    # rao = cpt.post_pro.rao(dataset, wave_direction=wave_direction)

    incoming_waves_elevation = airy_waves_free_surface_elevation(fs, diffraction_result)
    diffraction_elevation = solver.compute_free_surface_elevation(fs, diffraction_result)

    radiation_elevations_per_dof = {res.radiating_dof: solver.compute_free_surface_elevation(fs, res) for res in radiation_results}
    radiation_elevation = sum(rao[n] * radiation_elevations_per_dof[dof] for n, dof in enumerate(cal_body.dofs))

    # Animation.add_rotor = classmethod(add_rotor)

    r_axis = Axis([1,0,0], [0,0,hHub])

    ani_motion = sum(eta*rao[n] * ani_body.dofs[dof] for n, dof in enumerate(cal_body.dofs))
    rotor_motion = sum(eta*rao[n] * rotor_body.dofs[dof] for n, dof in enumerate(cal_body.dofs))

    animation:Animation
    animation = Animation.add_rotor(rotor_body, rotor_motion, loop_duration=2*np.pi/omega, axis=r_axis)

    animation.add_body(ani_body, faces_motion=ani_motion)

    animation.add_free_surface(fs, np.deg2rad(heading) * (incoming_waves_elevation*5 + diffraction_elevation*5 + radiation_elevation*5))
    # animation.add_body(ani_body, faces_motion=eta*rao_faces_motion)
    # animation.add_rotor(rotor_body, rotor_motion)
    animation.run(camera_position=(-225,-140,250))

    # fse = solver.get_free_surface_elevation(result=results, free_surface=ws)

    # anim = body_anim.animate(motion={"Surge": 0j}, loop_duration=1.0)
    # anim.add_free_surface(ws, fse+incoming_fse)
    # anim.run(camera_position=(-200,-200,200))

    animation.save(outFile, camera_position=(-576.5493790130992, -364.48192594934716, 380.27822555348655), nb_loops=2)

@classmethod
def add_rotor(cls, body, faces_motion=None, axis=None, faces_colors=None, edges=False, fps=24, loop_duration=2*np.pi):
    
    from capytaine.ui.vtk.helpers import compute_node_data, compute_vtk_polydata
    from capytaine.tools.optional_imports import import_optional_dependency
    
    vtk = import_optional_dependency("vtk")

    instance = cls(fps=fps, loop_duration=loop_duration)

    def add_rotor_actor(instance, body:cpt.FloatingBody, mesh:cpt.Mesh, axis:cpt.Axis, faces_motion=None, faces_colors=None, edges=False):

        """Add an animated object to the scene."""
        if faces_motion is not None:
            nodes_motion = compute_node_data(mesh.merged(), faces_motion)
            rotation_motion = np.zeros_like(faces_motion)

            nodes_rotation_motion = compute_node_data(mesh.merged(), rotation_motion)

        else:
            nodes_motion = None
    
        base_polydata = compute_vtk_polydata(mesh.merged())
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(base_polydata)
        actor = vtk.vtkActor()

        if edges:
            actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetInterpolationToGouraud()
        actor.SetMapper(mapper)

        if nodes_motion is not None or faces_colors is not None:
            instance._precomputed_polydatas[actor] = []
            for i_frame in range(instance.frames_per_loop):
                new_polydata = vtk.vtkPolyData()
                new_polydata.DeepCopy(base_polydata)
                if nodes_motion is not None:
                    # Change points positions at frame i
                    current_deformation = (
                            np.abs(nodes_motion)*np.cos(np.angle(nodes_motion)-2*np.pi*i_frame/instance.frames_per_loop)
                    )
                    rotated_body = body.copy()
                    rotated_body.rotate(axis, 2*np.pi*i_frame/instance.frames_per_loop)
                    # rotated_body.show()
                    # rotation_motion = sum([1 * rotated_body.dofs[dof] for n, dof in enumerate(rotated_body.dofs)])
                                    # - sum([1 * body.dofs[dof] for n, dof in enumerate(body.dofs)])
                    # nodes_rotation_motion = compute_node_data(rotated_body.mesh.merged(), rotation_motion)

                    points = new_polydata.GetPoints()
                    for j in range(mesh.nb_vertices):
                        point = points.GetPoint(j)
                        point = np.asarray(rotated_body.mesh.vertices[j]) + current_deformation[j]
                        points.SetPoint(j, tuple(point))
                if faces_colors is not None:
                    # Evaluate scalar field at frame i
                    current_colors = (
                        np.abs(faces_colors)*np.cos(np.angle(faces_colors)-2*np.pi*i_frame/instance.frames_per_loop)
                    )
                    max_val = max(abs(faces_colors))
                    vtk_faces_colors = vtk.vtkFloatArray()
                    for i, color in enumerate(current_colors):
                        vtk_faces_colors.InsertValue(i, (color+max_val)/(2*max_val))
                    new_polydata.GetCellData().SetScalars(vtk_faces_colors)
                instance._precomputed_polydatas[actor].append(new_polydata)
        else:
            instance._precomputed_polydatas[actor] = None
        instance.actors.append(actor)


        return actor
    
    actor = add_rotor_actor(instance=instance, body=body, mesh=body.mesh.merged(), axis=axis, faces_motion=faces_motion,
                               faces_colors=faces_colors, edges=edges)
    if faces_colors is None:
        actor.GetProperty().SetColor((0.9, 0.9, 0.9))
    else:
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(50)
        lut.SetHueRange(0, 0.6)
        lut.SetSaturationRange(0.5, 0.5)
        lut.SetValueRange(0.8, 0.8)
        lut.Build()
        actor.GetMapper().SetLookupTable(lut)

    return instance


####################################################################################################################
if __name__ == "__main__":

    solver = BEMSolver(memList=None)
    solver.getResultsFromWamit('BEM/OC4-semi-WAMIT', 'platform')
    solver.plotRadiationResults()
    solver.plotFexcResults(nHeading=1)
    
    solver = CapytaineSolver(memList=None, cog=[0,0,0], include0andinf=True, headings=[0, 30, 60, 90, ], nw=81)
    solver.getCapytaineMeshFromFile("models/nemoh/OC4_3087.dat")

    solver.defineProblems()
    solver.solveProblems(nThreads=8)

    solver.postProcess()
    solver.plotRadiationResults()
    solver.plotFexcResults()

    solver.writeWamit("tmp")
    solver = BEMSolver(memList=None)
    # readWamit1('tmp/Platform.1')
    # readWamit3('tmp/Platform.3')
    solver.getResultsFromWamit('tmp', 'Platform')

    solver.plotFexcResults()

    # solver.plotRadiationResults()


    # solver.defineProblems()
    # solver.solveProblems(nThreads=8)

    # solver.postProcess()

    # solver.writeWamit("tmp")

    # solver.plotRadiationResults()

    # solver.plotFexcResults()

    # animation()
    # dataset_tmp = solver.dataset0andInf.assign_coords(water_depth = solver.waterDepth, )

    # merged_dataset = xr.merge([solver.dataset0andInf, solver.dataset], compat='override')

    a = 1

    # solver.dataset.to_dataframe().to_csv("tmp.csv")

    a = 1