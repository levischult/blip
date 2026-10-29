import numpy as np
import jax.numpy as jnp
import jax
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import initialize_model
#from numpyro.contrib.nested_sampling import NestedSampler
from numpyro.distributions import constraints
import numpyro.distributions as dist
import pickle, dill
import os
import shutil


def numpyro_model(Model):
    '''
    Wrapper to translate our unified prior and likelihood to Numpyro-friendly input.
    
    Arguments
    -------------
    parameters (list of str)    : Model parameter name list
    prior_transform (function) : prior transform from unit cube
    log_likelihood (function) : desired log likelihood function, should take in theta parameter vector (we will do live adaptation to numpyro format below)
    '''
    
    with numpyro.plate("Npar",Model.Npar):
        theta = numpyro.sample('theta',dist.Uniform(0,1)) 
    
    theta_trans = numpyro.deterministic("theta_transformed", Model.prior(theta))
    
    numpyro.factor('loglike',log_factor=Model.likelihood(theta_trans))


def numpyro_model_sph(Model):
    '''
    Wrapper to translate our unified prior and likelihood to Numpyro-friendly input.
    
    The _sph version here accounts for the fact that the phase parameter prior should be periodic.
    
    Arguments
    -------------
    parameters (list of str)    : Model parameter name list
    prior_transform (function) : prior transform from unit cube
    log_likelihood (function) : desired log likelihood function, should take in theta parameter vector (we will do live adaptation to numpyro format below)
    '''
    Npar_c = len(Model.blm_phase_idx)
    Npar_nc = Model.Npar - Npar_c
    
    with numpyro.plate("Npar_nc",Npar_nc):
        theta_nc = numpyro.sample('theta_nc',dist.Uniform(0,1))
#        theta = numpyro.sample('theta',dist.ImproperUniform(constraints.real,(),()))
    with numpyro.plate("Npar_c",Npar_c):
        theta_c = numpyro.sample('theta_c',dist.ImproperUniform(constraints.real,(),()))
    
    theta_tot = theta_nc
    
    cnt = 0
    for idx in Model.blm_phase_idx:
        theta_tot = jnp.insert(theta_tot,idx,theta_c[cnt])
        cnt += 1
    
    theta = numpyro.deterministic("theta", theta_tot)
    
    theta_trans = numpyro.deterministic("theta_transformed", Model.prior(theta))
    
    numpyro.factor('loglike',log_factor=Model.likelihood(theta_trans))


class numpyro_engine():

    '''
    Class for interfacing with the Numpyro HMC (NUTS) sampler. 
    '''
    
    @classmethod
    def define_engine(cls, lisaobj, Nburn, Nsamples, Nthreads, prog, seed, gpu=False):
        
        ## multithreading setup
        ## default to parallel chains
        chain_method = 'parallel'
        if gpu:
            ## check number of available gpus
            N_GPU = jax.local_device_count(backend='gpu')
            if N_GPU == 1:
                if Nthreads > 1:
                    print("Nthreads = {}, but only one GPU is available. Setting numpyro chain_method to 'vectorized'.".format(Nthreads))
                    print(" WARNING: Vectorized GPU sampling is an experimental feature and is not stable for all BLIP configurations. If you get an XLA GEMM error, this is probably the cause; revert to standard parallelization in such cases.")
                    chain_method = 'vectorized'
            elif N_GPU > 1:
                if Nthreads > N_GPU:
                    print("Nthreads ({}) > N_GPU ({}) but vectorized parallel sampling has not yet been implemented. Setting Nthreads = N_GPU.".format(Nthreads,N_GPU))
                    Nthreads = N_GPU
            else:
                raise ValueError("GPU usage was requested but no GPUs are available!")
        else:
            numpyro.set_host_device_count(Nthreads)
        
        
        if seed is not None:
            rng_key = jax.random.PRNGKey(seed)
        else:
            raise TypeError("Numpyro sampler requires a defined seed.")
        
        ## if there are phase parameters, use the sph wrapper
        if len(lisaobj.Model.blm_phase_idx) > 0:
            kernel = NUTS(numpyro_model_sph)
        ## otherwise use the standard one
        else:
            kernel = NUTS(numpyro_model)
        
        engine = MCMC(kernel,num_warmup=Nburn,num_samples=Nsamples,num_chains=Nthreads,chain_method=chain_method,progress_bar=prog)

        # print npar
        print("Npar = " + str(lisaobj.Model.Npar))

        return engine, lisaobj.Model.parameters, rng_key

    @staticmethod
    def find_map(engine, lisaModel, rng_key, Nsteps, lr):
        '''

        Optimize each chain to a maximum a posteriori point, for use as the NUTS starting point.
        The objective is numpyro's potential function in unconstrained space, i.e. exactly the density
        NUTS samples (including the Jacobian of the unit-cube -> real transform), so the optimum is always
        interior to the prior bounds. Each chain is optimized independently from its own random start.

        Arguments
        ------------------------------
        engine : numpyro.infer.MCMC
        	MCMC engine from define_engine; supplies the model function and number of chains.
        lisaModel : Model
        	BLIP Model object passed to the numpyro model as its argument.
        rng_key : jax.random.PRNGKey
        	Key used to draw the per-chain random starting points.
        Nsteps : int
        	Number of Adam steps per chain.
        lr : float
        	Adam learning rate (step size in unconstrained space).

        Returns
        ------------------------------
        init_params : dict
        	Per-chain optimized latent values in unconstrained space, keyed by sample site, with a leading
        	chain dimension when there is more than one chain. Can be passed directly to MCMC.run/warmup.
        map_samples : numpy.ndarray
        	Optimized transformed (physical) parameters, shape (Nchains, Npar).

        '''
        Nchains = engine.num_chains

        ## one random start per chain via numpyro's default init_to_uniform (U[-2,2] in unconstrained space)
        ## batched keys give z0 with a leading (Nchains,) dim; potential_fn/postprocess_fn act on a single chain
        model_info = initialize_model(jax.random.split(rng_key,Nchains),engine.sampler.model,model_args=(lisaModel,))
        z0 = model_info.param_info.z
        potential_fn = model_info.potential_fn

        optimizer = numpyro.optim.Adam(step_size=lr)

        def optimize_chain(z_init):
            def adam_step(i, opt_state):
                grads = jax.grad(potential_fn)(optimizer.get_params(opt_state))
                return optimizer.update(grads,opt_state)
            opt_state = jax.lax.fori_loop(0,Nsteps,adam_step,optimizer.init(z_init))
            z_opt = optimizer.get_params(opt_state)
            return z_opt, potential_fn(z_opt)

        ## lax.map runs chains sequentially under a single compilation, so memory use stays at one likelihood
        ## evaluation (vmap could exhaust GPU memory for large models)
        print("Optimizing to MAP estimate for initialization ({} chain(s), {} steps each)...".format(Nchains,Nsteps))
        z_map, pe_map = jax.jit(lambda z: jax.lax.map(optimize_chain,z))(z0)
        pe_map = np.asarray(pe_map)

        ## fall back to the random start for any chain whose optimization diverged
        failed = ~np.isfinite(pe_map)
        if np.any(failed):
            print("WARNING: MAP optimization gave a non-finite log density for chain(s) {}. Reverting these chains to their random starting points.".format(np.flatnonzero(failed)))
            z_map = jax.tree.map(lambda zm, zi: jnp.where(failed.reshape((-1,)+(1,)*(zm.ndim-1)),zi,zm),z_map,z0)
        for i in range(Nchains):
            print("    Chain {}: log posterior density at MAP = {}".format(i,-pe_map[i]))

        ## transformed (physical) parameters at the optimum; theta_transformed is a per-parameter list,
        ## so transpose to (Nchains, Npar) to match the post_samples convention
        map_samples = np.asarray(jax.vmap(model_info.postprocess_fn)(z_map)['theta_transformed']).T

        ## numpyro expects no chain dim for a single chain
        if Nchains == 1:
            z_map = jax.tree.map(lambda x: x[0],z_map)

        return z_map, map_samples

    @staticmethod
    def run_engine(engine,lisaobj,rng_key,init_params=None):

        # -------------------- Run HMC sampler ---------------------------
        print("Beginning sampling...")
        ## init_params (unconstrained, e.g. from find_map) overrides numpyro's default initialization if given
        engine.run(rng_key,lisaobj.Model,init_params=init_params)
        print("Sampling complete. Retrieving posterior and plotting results...")
        ## retrive samples and reformat
        post_samples = np.array(engine.get_samples()['theta_transformed']).T
        
        return post_samples
    
    
    def load_engine(resume_file):
        
        ## load model and parameters from previous checkpoint
        if os.path.isfile(resume_file):
            print("Loading interrupted analysis from last checkpoint...")
            with open(resume_file,'rb') as file:
                engine,state,chain = pickle.load(file)
            ## tell numpyro to start from current state
            engine.post_warmup_state = state
            ## grab rng_key for running
            rng_key = engine.post_warmup_state.rng_key
        else:
            raise TypeError("Checkpoint file <{}> does not exist. Cannot resume from checkpoint.".format(resume_file))
    
        return engine, rng_key, chain
    
    
    @staticmethod
    def run_engine_with_checkpointing(engine,lisaModel,rng_key,chain,checkpoint_file,Ntotal,checkpoint_at,resume=False,init_params=None):
        '''
        Runs the numpyro sampler with sampler state checkpointing. 
        
        Arguments
        -------------------
        [fill in]
        checkpoint_sampling (bool)  : When to checkpoint. Options:
                                        'end' (only saves sampler state at the very end of the run)
                                        'warmup' (saves after warmup phase and at end)
                                        'interval' (saves after warmup, at end, and after every checkpoint_interval number of samples)
                                        Note: Generally not worth checkpointing while sampling for large models/datasets, 
                                        as the recompliation and GPU off/onloading time will exceed the sampling time.

        resume (bool)               : Whether the run is being resumed. If so, skip the warmup phase.
        init_params (dict)          : Initial latent values in unconstrained space (e.g. from find_map). Only used
                                        by the first warmup/run call; ignored once a post-warmup state exists.
                                        If None, use numpyro's default initialization.
        Returns
        --------------------
        post_samples (array)        : Posterior samples.
        
        '''
        

        if chain is None and not resume:
            if checkpoint_at=='warmup' or checkpoint_at=='interval':
                print("Beginning sampling, starting warmup phase...")
                ## run warmup phase
                engine.warmup(rng_key,lisaModel,init_params=init_params)
                ## initialization is now carried by the post-warmup state
                init_params = None
                state = engine.post_warmup_state
                print("Warmup phase complete. Checkpointing before initializing sampling...")
                ## save
                if dill.pickles([engine,state,chain]):
                    temp_file = checkpoint_file + ".temp"
                    with open(temp_file, "wb") as file:
                        pickle.dump([engine,state,chain], file)
                    shutil.move(temp_file, checkpoint_file)
                else:
                    print("WARNING: Cannot write checkpoint file, job cannot resume if interrupted.")
                
                engine.post_warmup_state = state
                rng_key = engine.post_warmup_state.rng_key
            ## warn if the checkpointing spec is wonky, but continue on as if it were 'end'
            elif checkpoint_at!='end':
                print("WARNING: Invalid specification of checkpointing behavior (checkpoint_at='{}'). Sampler state will be saved at end of sampling.".format(checkpoint_at))
        
        print("Initializing sampling...")
        while True:
            
            engine.run(rng_key,lisaModel,init_params=init_params)
            init_params = None

            ## get state, current chain
            state = engine.last_state
            chain_update = engine.get_samples()
            if chain is not None:
                chain_updated = {}
                chain_updated['theta'] = jnp.append(chain['theta'],chain_update['theta'],axis=0)
                chain_updated['theta_transformed'] = [jnp.append(chain['theta_transformed'][i],chain_update['theta_transformed'][i]) for i in range(len(chain_update['theta_transformed']))]
                chain = chain_updated
            else:
                chain = chain_update
            
            if checkpoint_at=='interval':
                ## check to see if we have the desired number of samples yet
                Ncurrent = len(chain['theta_transformed'][0])
                if Ncurrent >= Ntotal:
                    break
            else:
                break

            print("Checkpointing ({}/{} samples)...".format(Ncurrent,Ntotal))
            ## save
            if dill.pickles([engine,state,chain]):
                temp_file = checkpoint_file + ".temp"
                with open(temp_file, "wb") as file:
                    pickle.dump([engine,state,chain], file)
                shutil.move(temp_file, checkpoint_file)
            else:
                print("Warning: Cannot write checkpoint file, job cannot resume if interrupted.")
            
            ## tell numpyro to start from current state
            engine.post_warmup_state = state
            ## grab rng_key for running
            rng_key = engine.post_warmup_state.rng_key
        
        ## save the final state
        print("Sampling complete. Saving final sampler state to {}".format(checkpoint_file))
        if dill.pickles([engine,state,chain]):
            temp_file = checkpoint_file + ".temp"
            with open(temp_file, "wb") as file:
                pickle.dump([engine,state,chain], file)
            shutil.move(temp_file, checkpoint_file)
        else:
            print("Warning: Failed to save final state to checkpoint file, cannot resume sampling later.")
    
        ## retrive samples and reformat
        post_samples = np.array(chain['theta_transformed']).T
        
        return post_samples


#class numpyro_nested_engine():
#
#    '''
#    Class for interfacing with the Numpyro nested sampling sampler. 
#    '''
#    
#    @classmethod
#    def define_engine(cls, lisaobj, Nsamples, seed, gpu=False):
#        
#        if seed is not None:
#            rng_key = jax.random.PRNGKey(seed)
#        else:
#            raise TypeError("Numpyro sampler requires a defined seed.")
#        
#        constructor_kwargs = {'verbose':True,'parameter_estimation':True, 'num_live_points':800, 'max_samples':10*Nsamples}
#        termination_kwargs = {'dlogZ':1e-4}
#        ## if there are phase parameters, use the sph wrapper
#        if len(lisaobj.Model.blm_phase_idx) > 0:
#            engine = NestedSampler(numpyro_model_sph, constructor_kwargs=constructor_kwargs, termination_kwargs=termination_kwargs)
#        ## otherwise use the standard one
#        else:
#            engine = NestedSampler(numpyro_model, constructor_kwargs=constructor_kwargs, termination_kwargs=termination_kwargs)
#        engine.num_samples = Nsamples
##        engine = MCMC(kernel,num_warmup=Nburn,num_samples=Nsamples,num_chains=Nthreads,chain_method=chain_method,progress_bar=prog)
#
#        # print npar
#        print("Npar = " + str(lisaobj.Model.Npar))
#
#        return engine, lisaobj.Model.parameters, rng_key
#    
#    @staticmethod
#    def run_engine(engine,lisaModel,rng_key,checkpoint_file):
#        
#        # -------------------- Run HMC sampler ---------------------------
#        print("Beginning sampling...")
#        engine.run(rng_key,lisaModel)
#        print("Sampling complete. Retrieving posterior and plotting results...")
#        ## retrive samples, resample from weighted samples, and reformat
#        _, new_key = jax.random.split(rng_key,2)
#        post_samples = np.array(engine.get_samples(new_key,num_samples=engine.num_samples)['theta_transformed']).T
#        
#        print("Sampling complete. Saving final sampler state to {}".format(checkpoint_file))
#        if dill.pickles(engine):
#            temp_file = checkpoint_file + ".temp"
#            with open(temp_file, "wb") as file:
#                pickle.dump(engine, file)
#            shutil.move(temp_file, checkpoint_file)
#        else:
#            print("Warning: Failed to save final state to checkpoint file, cannot resume sampling later.")
#    
#        
#        return post_samples
#    
#    
#    def load_engine(resume_file):
#        
#        ## load model and parameters from previous checkpoint
#        if os.path.isfile(resume_file):
#            print("Loading interrupted analysis from last checkpoint...")
#            with open(resume_file,'rb') as file:
#                engine = pickle.load(file)
#        else:
#            raise TypeError("Checkpoint file <{}> does not exist. Cannot resume from checkpoint.".format(resume_file))
#    
#        return engine
#    