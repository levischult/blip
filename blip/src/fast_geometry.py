import numpy as np
import jax.numpy.linalg as LA
#from scipy.special import lpmn, sph_harm
from multiprocessing import Pool
from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import healpy as hp
from scipy.special import sph_harm
from blip.src.sph_geometry import sph_geometry

class geometry(sph_geometry):

    '''
    Module containing geometry methods. The methods here include calculation of antenna patters for a single doppler channel, for the three michelson channels or for the AET TDI channels and calculation of noise power spectra for various channel combinations.
    '''

    def __init__(self):

        if (not self.injection and self.params['sph_flag']) or (self.injection and self.inj['sph_flag']):
        
#        if self.params['sph_flag'] or self.inj['sph_flag']:
            sph_geometry.__init__(self)



    def lisa_orbits(self, tsegmid):


        '''
        Define LISA orbital positions at the midpoint of each time integration segment using analytic MLDC orbits.

        Parameters
        -----------

        tsegmid  :  array
            A numpy array of the tsegmid for each time integration segment.

        Returns
        -----------
        rs1, rs2, rs3  :  array
            Arrays of satellite positions for each segment midpoint in timearray. e.g. rs1[1] is [x1,y1,z1] at t=midpoint[1]=timearray[1]+(segment length)/2.
        '''
        ## Branch orbiting and stationary cases; compute satellite position in stationary case based off of first time entry in data.
        if self.params['lisa_config'] == 'stationary':
            # Calculate start time from tsegmid
            tstart = tsegmid[0] - (tsegmid[1] - tsegmid[0])/2
            # Fill times array with just the start time
            times = np.empty(len(tsegmid))
            times.fill(tstart)
        elif self.params['lisa_config'] == 'orbiting':
            times = tsegmid
        else:
            raise ValueError('Unknown LISA configuration selected')


        ## Semimajor axis in m
        a = 1.496e11


        ## Alpha and beta phases allow for changing of initial satellite orbital phases; default initial conditions are alphaphase=betaphase=0.
        betaphase = 0
        alphaphase = 0

        ## Orbital angle alpha(t)
        at = (2*np.pi/31557600)*times + alphaphase

        ## Eccentricity. L-dependent, so needs to be altered for time-varied arm length case.
        e = self.armlength/(2*a*np.sqrt(3))

        ## Initialize arrays
        beta_n = (2/3)*np.pi*np.array([0,1,2])+betaphase

        ## meshgrid arrays
        Beta_n, Alpha_t = np.meshgrid(beta_n, at)

        ## Calculate inclination and positions for each satellite.
        x_n = a*np.cos(Alpha_t) + a*e*(np.sin(Alpha_t)*np.cos(Alpha_t)*np.sin(Beta_n) - (1+(np.sin(Alpha_t))**2)*np.cos(Beta_n))
        y_n = a*np.sin(Alpha_t) + a*e*(np.sin(Alpha_t)*np.cos(Alpha_t)*np.cos(Beta_n) - (1+(np.cos(Alpha_t))**2)*np.sin(Beta_n))
        z_n = -np.sqrt(3)*a*e*np.cos(Alpha_t - Beta_n)


        ## Construct position vectors r_n
        rs1 = np.array([x_n[:, 0],y_n[:, 0],z_n[:, 0]])
        rs2 = np.array([x_n[:, 1],y_n[:, 1],z_n[:, 1]])
        rs3 = np.array([x_n[:, 2],y_n[:, 2],z_n[:, 2]])

        return rs1, rs2, rs3

    ###################################################################################
    ## Wrapper functions to return the corresponding response array from its entries ##
    ###################################################################################
    
    def isgwb_wrapper(self,F1_ii, F2_ii, F3_ii, F12_ii, F13_ii, F23_ii):
        '''
        Wrapper function to take sky integral and return response array slice
        '''
        R1_ii  = self.dOmega/(4*np.pi)*np.sum(F1_ii, axis=1 )
        R2_ii  = self.dOmega/(4*np.pi)*np.sum(F2_ii, axis=1 ) 
        R3_ii  = self.dOmega/(4*np.pi)*np.sum(F3_ii, axis=1 ) 
        R12_ii = self.dOmega/(4*np.pi)*np.sum(F12_ii, axis=1) 
        R13_ii = self.dOmega/(4*np.pi)*np.sum(F13_ii, axis=1) 
        R23_ii = self.dOmega/(4*np.pi)*np.sum(F23_ii, axis=1) 
        
        return np.array([ [R1_ii, R12_ii, R13_ii] , [np.conj(R12_ii), R2_ii, R23_ii], [np.conj(R13_ii), np.conj(R23_ii), R3_ii] ])
    
    def sph_asgwb_wrapper(self,F1_ii, F2_ii, F3_ii, F12_ii, F13_ii, F23_ii):
        '''
        Wrapper function to convolve with Ylms and return response array slice
        '''
        R1_ii  = self.dOmega*np.einsum('ij, jk', F1_ii, self.Ylms)
        R2_ii  = self.dOmega*np.einsum('ij, jk', F2_ii, self.Ylms)
        R3_ii = self.dOmega*np.einsum('ij, jk', F3_ii, self.Ylms)
        R12_ii = self.dOmega*np.einsum('ij, jk', F12_ii, self.Ylms)
        R13_ii = self.dOmega*np.einsum('ij, jk', F13_ii, self.Ylms)
        R23_ii = self.dOmega*np.einsum('ij, jk', F23_ii, self.Ylms)
        R21_ii = self.dOmega*np.einsum('ij, jk', np.conj(F12_ii), self.Ylms)
        R31_ii = self.dOmega*np.einsum('ij, jk', np.conj(F13_ii), self.Ylms)
        R32_ii = self.dOmega*np.einsum('ij, jk', np.conj(F23_ii), self.Ylms)
    
        return np.array([ [R1_ii, R12_ii, R13_ii] , [R21_ii, R2_ii, R23_ii], [R31_ii, R32_ii, R3_ii] ])

    
    def pix_convolved_asgwb_wrapper(self,F1_ii, F2_ii, F3_ii, F12_ii, F13_ii, F23_ii,skymap=None):
        '''
        Wrapper function to convolve with pixel-basis skymap and return response array slice
        '''
        ## Detector response summed over polarization and integrated over sky direction
        ## The travel time phases for the which are relevent for the cross-channel are
        ## accounted for in the Fplus and Fcross expressions above.
        R1_ii  = self.dOmega*np.sum( F1_ii * skymap[None, :], axis=1 )
        R2_ii  = self.dOmega*np.sum( F2_ii * skymap[None, :], axis=1 ) 
        R3_ii  = self.dOmega*np.sum( F3_ii * skymap[None, :], axis=1 ) 
        R12_ii = self.dOmega*np.sum( F12_ii * skymap[None, :], axis=1) 
        R13_ii = self.dOmega*np.sum( F13_ii * skymap[None, :], axis=1) 
        R23_ii = self.dOmega*np.sum( F23_ii * skymap[None, :], axis=1) 
        
        return np.array([ [R1_ii, R12_ii, R13_ii] , [np.conj(R12_ii), R2_ii, R23_ii], [np.conj(R13_ii), np.conj(R23_ii), R3_ii] ])
    
    def pix_unconvolved_asgwb_wrapper(self,F1_ii, F2_ii, F3_ii, F12_ii, F13_ii, F23_ii):
        '''
        Wrapper function which just returns its inputs, as the unconvolved case doesn't integrate over the sky
        '''
        return np.array([ [F1_ii, F12_ii, F13_ii] , [np.conj(F12_ii), F2_ii, F23_ii], [np.conj(F13_ii), np.conj(F23_ii), F3_ii] ])
    
    
    ########################################################################
    ## For unpacking per-frequency responses into the correct array shape ##
    ########################################################################
    
    def unpack_wrapper_33ft(self,sm,ii,Rf):
        '''
        Wrapper function for unpacking responses of shape 3 x 3 x frequency x time
        '''
        sm.response_mat[:,:,ii,:] = Rf
        return
    
    def unpack_wrapper_33ftn(self,sm,ii,Rf):
        '''
        Wrapper function for unpacking responses of shape 3 x 3 x frequency x time x spatial
        
        Spatial can mean either spherical harmonics or pixels.
        '''
        sm.response_mat[:,:,ii,:,:] = Rf
        return
    
    
    
    
    
    ##########################################
    ## For parallelizing across frequencies ##
    ##########################################

    def frequency_response_wrapper(self,ii):
        
        '''
        Wrapper function to help with parallelization of the response function calculations.
        
        Arguments
        
        ii (int)   :  Frequency index
        
        Returns
        
        response_ii : Response matrix in that frequency bin
        '''
        
        # Calculate GW transfer function for the michelson channels
        gammaU_plus    =    1/2 * (jnp.sinc((self.f0[ii])*(1 - self.udir)/np.pi)*jnp.exp(-1j*self.f0[ii]*(3+self.udir)) + \
                         jnp.sinc((self.f0[ii])*(1 + self.udir)/np.pi)*jnp.exp(-1j*self.f0[ii]*(1+self.udir)))

        gammaV_plus    =    1/2 * (jnp.sinc((self.f0[ii])*(1 - self.vdir)/np.pi)*jnp.exp(-1j*self.f0[ii]*(3+self.vdir)) + \
                         jnp.sinc((self.f0[ii])*(1 + self.vdir)/np.pi)*jnp.exp(-1j*self.f0[ii]*(1+self.vdir)))

        gammaW_plus    =    1/2 * (jnp.sinc((self.f0[ii])*(1 - self.wdir)/np.pi)*jnp.exp(-1j*self.f0[ii]*(3+self.wdir)) + \
                         jnp.sinc((self.f0[ii])*(1 + self.wdir)/np.pi)*jnp.exp(-1j*self.f0[ii]*(1+self.wdir)))


        # Calculate GW transfer function for the michelson channels
        gammaU_minus    =    1/2 * (jnp.sinc((self.f0[ii])*(1 + self.udir)/np.pi)*jnp.exp(-1j*self.f0[ii]*(3 - self.udir)) + \
                         jnp.sinc((self.f0[ii])*(1 - self.udir)/np.pi)*jnp.exp(-1j*self.f0[ii]*(1 - self.udir)))

        gammaV_minus    =    1/2 * (jnp.sinc((self.f0[ii])*(1 + self.vdir)/np.pi)*jnp.exp(-1j*self.f0[ii]*(3 - self.vdir)) + \
                         jnp.sinc((self.f0[ii])*(1 - self.vdir)/np.pi)*jnp.exp(-1j*self.f0[ii]*(1 - self.vdir)))

        gammaW_minus    =    1/2 * (jnp.sinc((self.f0[ii])*(1 + self.wdir)/np.pi)*jnp.exp(-1j*self.f0[ii]*(3 - self.wdir)) + \
                         jnp.sinc((self.f0[ii])*(1 - self.wdir)/np.pi)*jnp.exp(-1j*self.f0[ii]*(1 - self.wdir)))


        ## Michelson antenna patterns
        ## Calculate Fplus
        Fplus1 = 0.5*(self.Fplus_u*gammaU_plus - self.Fplus_v*gammaV_plus)*jnp.exp(-1j*self.f0[ii]*(self.udir + self.vdir)/np.sqrt(3))
        Fplus2 = 0.5*(self.Fplus_w*gammaW_plus - self.Fplus_u*gammaU_minus)*jnp.exp(-1j*self.f0[ii]*(-self.udir + self.vdir)/np.sqrt(3))
        Fplus3 = 0.5*(self.Fplus_v*gammaV_minus - self.Fplus_w*gammaW_minus)*jnp.exp(1j*self.f0[ii]*(self.vdir + self.wdir)/np.sqrt(3))

        ## Calculate Fcross
        Fcross1 = 0.5*(self.Fcross_u*gammaU_plus  - self.Fcross_v*gammaV_plus)*jnp.exp(-1j*self.f0[ii]*(self.udir + self.vdir)/np.sqrt(3))
        Fcross2 = 0.5*(self.Fcross_w*gammaW_plus  - self.Fcross_u*gammaU_minus)*jnp.exp(-1j*self.f0[ii]*(-self.udir + self.vdir)/np.sqrt(3))
        Fcross3 = 0.5*(self.Fcross_v*gammaV_minus - self.Fcross_w*gammaW_minus)*jnp.exp(1j*self.f0[ii]*(self.vdir + self.wdir)/np.sqrt(3))

        ## Detector response averaged over polarization
        ## The travel time phases for the which are relevent for the cross-channel are
        ## accounted for in the Fplus and Fcross expressions above.
        F1_ii  = (1/2)*((jnp.absolute(Fplus1))**2 + (jnp.absolute(Fcross1))**2)
        F2_ii  = (1/2)*((jnp.absolute(Fplus2))**2 + (jnp.absolute(Fcross2))**2) 
        F3_ii  = (1/2)*((jnp.absolute(Fplus3))**2 + (jnp.absolute(Fcross3))**2) 
        F12_ii = (1/2)*(jnp.conj(Fplus1)*Fplus2 + jnp.conj(Fcross1)*Fcross2)
        F13_ii = (1/2)*(jnp.conj(Fplus1)*Fplus3 + jnp.conj(Fcross1)*Fcross3)
        F23_ii = (1/2)*(jnp.conj(Fplus2)*Fplus3 + jnp.conj(Fcross2)*Fcross3)
        
        sm_response_slices = []
        for sm in self.submodels:
            sm_response_slices.append(sm.response_wrapper_func(F1_ii, F2_ii, F3_ii, F12_ii, F13_ii, F23_ii,**sm.response_kwargs))
        
        return sm_response_slices
        
        ## return the frequency response components sans alteration
#        return R1_ii, R2_ii, R3_ii, R12_ii, R13_ii, R23_ii
        
#        return np.array([ [R1_ii, R12_ii, R13_ii] , [np.conj(R12_ii), R2_ii, R23_ii], [np.conj(R13_ii), np.conj(R23_ii), R3_ii] ])


    def get_geometric_Fplus_Fcross(self,arm_hat_ij,mhat_prod,nhat_prod):
        '''
        Wrapper function to reduce repeated calculations for the geometric components of the antenna pattern functions
        
        Arguments
        -------------------
        arm_hat_ij (float arrays) : directional unit vector for the arm between satellites i and j.
        mhat_prod, nhat_prod (float arrays) : outer products of phi hat and theta hat, respectively
        
        Returns
        --------------------------
        Fplus_ij, Fcross_ij (float arrays) : Geometric components of the plus and cross antenna pattern functions for LISA arms i and j
        '''
        
        interior_prod = jnp.einsum("ik,jk -> ijk",arm_hat_ij, arm_hat_ij)
        
        return 0.5*jnp.einsum("ijk,ijl", interior_prod, mhat_prod - nhat_prod), 0.5*jnp.einsum("ijk,ijl", interior_prod, mhat_prod + nhat_prod)
    
    
    def generic_mich_response(self,f0,tsegmid,submodels):
        
        '''
        Prototype federated function for response function calculations, designed to eliminate all redundant calculations when computing the response functions for multiple submodels.
        
        
        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

        tsegmid  :  float
            A numpy array of segment midpoints
    
        submodels : list of instantiated submodel objs
        
        ? : whatever else

        Returns
        -----------
        
        response_mat(s) (numpy array/s) : the appropriate response function for each submodel, attached to that submodel
        
        '''
        
        ## basic preliminaries
        self.f0 = f0
        
        # Area of each pixel in sq.radians
        self.dOmega = hp.pixelfunc.nside2pixarea(self.params['nside'])
        
        ## let's do something like
        fullsky = np.any([sm.fullsky for sm in submodels if hasattr(sm,"fullsky")])
        
        # Make relevant array of pixel indices
        if fullsky:
            ## if any of the submodels require evaluation over the full sky, we need to do so
            npix = hp.nside2npix(self.params['nside'])
            pix_idx  = np.arange(npix)
        else:
            ## otherwise, make a map of everwhere on the sky where there is power across all submodels
            combined_map = np.sum([sm.skymap for sm in submodels if hasattr(sm,'skymap')]) ##THIS IS PSEUDOCODE, FILL IN WITH ACTUAL VARS
            ## Array of pixel indices where the combined map is nonzero
            pix_idx = np.flatnonzero(combined_map)

        # Angular coordinates of pixel indices
        theta, phi = hp.pix2ang(self.params['nside'], pix_idx)

        # Take cosine.
        ctheta = np.cos(theta)


        ################################################################################
        ## ode block to set up which submodels we have and any specifics they require ##
        ################################################################################
        for sm in submodels:
            ## instantiate the response array and specify additional wrappers for unpacking
            ## this is mapped to the response_wrapper_func attached to the submodel
            if sm.spatial_model_name == 'isgwb':
                sm.unpack_response = self.unpack_wrapper_33ft
                sm.response_shape = (3,3,f0.size,tsegmid.size)
            elif sm.basis == 'sph':
                sm.unpack_response = self.unpack_wrapper_33ftn
                ## array size of almax
                alm_size = (sm.almax + 1)**2
                ## initalize array for Ylms
                self.Ylms = np.zeros((npix, alm_size ), dtype='complex')
                ## Get the spherical harmonics
                for ii in range(alm_size):
                    lval, mval = self.idxtoalm(sm.almax, ii)
                    self.Ylms[:, ii] = sph_harm(mval, lval, phi, theta)
                sm.response_shape = (3,3,f0.size, tsegmid.size,alm_size)
            elif sm.basis == 'pixel':
                if hasattr(sm,"fixedmap") and sm.fixedmap:
                    sm.unpack_response = self.unpack_wrapper_33ft
                    sm.response_shape = (3,3,f0.size, tsegmid.size)
                else:
                    sm.unpack_response = self.unpack_wrapper_33ftn
                    sm.response_shape = (3,3,f0.size, tsegmid.size,pix_idx.size)
            else:
                raise ValueError("Specification of the response wrapper is unrecognized. Check the implementation of response_wrapper_func for submodel "+sm.spatial_model_name+" in models.py.")
        

        ##############################################################
        ## this will be the main bit which remains largely the same ##
        ##############################################################

        # Create 2D array of (x,y,z) unit vectors for every sky direction.
        omegahat = np.array([np.sqrt(1-ctheta**2)*np.cos(phi),np.sqrt(1-ctheta**2)*np.sin(phi),ctheta])

        # Call lisa_orbits to compute satellite positions at the midpoint of each time segment
        rs1, rs2, rs3 = self.lisa_orbits(tsegmid)
        
        ## get the unit vectors for each arm, as we need them frequently
        uhat_21 = (rs2-rs1)/LA.norm(rs2-rs1,axis=0)[None, :]
        vhat_31 = (rs3-rs1)/LA.norm(rs3-rs1,axis=0)[None, :]
        what_32 = (rs3-rs2)/LA.norm(rs3-rs2,axis=0)[None, :]
        
        ## Calculate directional unit vector dot products
        ## Dimensions of udir is time-segs x sky-pixels
        self.udir = np.einsum('ij,ik',uhat_21,omegahat)
        self.vdir = np.einsum('ij,ik',vhat_31,omegahat)
        self.wdir = np.einsum('ij,ik',what_32,omegahat)


        ## NB --    An attempt to directly adapt e.g. (u o u):e+ as implicit tensor calculations
        ##             as opposed to the explicit forms we've previously used. '''


        ## CHECK THESE EXPLANATIONS LATER AGAINST MY OLD NOTEBOOK
        ######################################################################
        ## mhat and nhat are phi hat and theta hat in cartesian coordinates ##
        ######################################################################
        
        mhat = np.array([np.sin(phi),-np.cos(phi),np.zeros(len(phi))])
        nhat = np.array([np.cos(phi)*ctheta,np.sin(phi)*ctheta,-np.sqrt(1-ctheta**2)])

        ## outer (self-) products of mhat and nhat
        mhat_op = jnp.einsum("ik,jk -> ijk",mhat,mhat)
        
        nhat_op = jnp.einsum("ik,jk -> ijk",nhat,nhat)
        
        # 1/2 u x u : eplus. These depend only on geometry so they only have a time and directionality dependence and not of frequency
        ## we have jitted this function
        
        
        ## commenting out some parallelization. Can reintroduce later, but these calculations are not egregious, so I don't think we'll gain much over 3 calculations.
#        self.arm_iterator = [uhat_21,vhat_31,what_32]
#        idx = range(len(arm_iterator))
#        with Pool(self.inj['response_nthread']) as pool:
#            result = pool.map(self.get_geometric_Fplus_Fcross,idx)
#        
#            for ii, R_f in zip(idx,result):
#                response_mat[:,:,ii,:] = R_f
#        
        
        self.Fplus_u, self.Fcross_u = self.get_geometric_Fplus_Fcross(uhat_21,mhat_op,nhat_op)
        self.Fplus_v, self.Fcross_v = self.get_geometric_Fplus_Fcross(vhat_31,mhat_op,nhat_op)
        self.Fplus_w, self.Fcross_w = self.get_geometric_Fplus_Fcross(what_32,mhat_op,nhat_op)


        ###############################################################################################################################################################
        ## left off here; this is where we'll call the frequency wrapper and take the appropriate sums/averages before constructing the relevant full response mats. ## 
        ###############################################################################################################################################################

        
        for sm in submodels:
            ## instantiate the response arrays
            sm.response_mat = np.zeros(sm.response_shape, dtype='complex') ## we can define what shape the final response should be in the submodel initialzation
        
        # Calculate the detector response for each frequency
        idx = range(0,f0.size)
        
        with Pool(self.inj['response_nthread']) as pool:
            result = pool.map(self.frequency_response_wrapper,idx)
            for sm in submodels:
                for ii, R_f in zip(idx,result):
                    ## index appropriately for each submodel
                    sm.unpack_response(sm,ii,R_f)

        return









































