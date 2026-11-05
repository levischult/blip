import numpy as np
from dynesty import NestedSampler
from dynesty.utils import resample_equal
import dill
import time
import shutil, os
import threading
from concurrent.futures import ThreadPoolExecutor, wait


###################################################
###   JAX LIKELIHOOD <-> DYNESTY TRANSLATION    ###
###################################################

## dynesty runs on the CPU; the BLIP prior + likelihood are evaluated as one jitted JAX function of the unit cube on
## the CPU or GPU (JaxLikelihood). With Nthreads > 1, dynesty's parallel proposals run in threads whose likelihood
## calls are coalesced into single batched device calls (BatchedLikelihoodPool).
## jax is imported lazily, inside JaxLikelihood, when the likelihood is first compiled.


def unit_cube_transform(u):
    '''
    Identity prior transform for use with JaxLikelihood, which applies the BLIP prior internally.

    Arguments
    -------------
    u (array) : draw from the unit cube

    Returns
    -------------
    u (array) : copy of the input
    '''
    return np.array(u)


def _next_pow2(n):
    '''Smallest power of two >= n.'''
    return 1 << (int(n) - 1).bit_length()


class JaxLikelihood():

    '''
    Picklable, jitted log likelihood of the unit cube for BLIP Models, for use with CPU-based samplers.

    Large Model arrays (data correlation matrix, response functions) are closed over and therefore baked into the
    compiled function as device constants, so only the Npar-length parameter vector crosses host <-> device per call.

    Arguments
    -------------
    Model (Model)          : BLIP Model object
    platform (str)         : 'gpu' or 'cpu'; the first device of this platform is used
    max_batch (int)        : largest batch the sampler will request (e.g. dynesty queue_size)
    batch_size (int)       : if not None, evaluate batches with jax.lax.map in chunks of this size to bound device memory.
                             If None, the whole batch is vmapped at once.
    '''

    ## attributes that hold compiled functions/devices/threads and must not be pickled
    _transient = ('_dev', '_device_put', '_f_single', '_f_batch', '_f_transform', '_batcher')

    def __init__(self, Model, platform='gpu', max_batch=1, batch_size=None):
        self.Model = Model
        self.Npar = Model.Npar
        self.platform = platform
        self.max_batch = max(int(max_batch), 1)
        self.batch_size = batch_size
        self._reset_transient()

    def _reset_transient(self):
        '''Clear all compiled/device/thread state; it is rebuilt lazily on the next call.'''
        for attr in self._transient:
            setattr(self, attr, None)

    def __getstate__(self):
        '''Drop compiled functions and the batching pool so that checkpoints only contain the Model.'''
        state = self.__dict__.copy()
        for attr in self._transient:
            state.pop(attr, None)
        return state

    def __setstate__(self, state):
        '''Restore from a checkpoint; compiled functions are rebuilt on first use.'''
        self.__dict__.update(state)
        self._reset_transient()

    def _build(self):
        '''Jit the fused prior + likelihood (single and batched) and the batched prior transform on the target device.'''
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        Model = self.Model

        ## same composition as numpyro_model: likelihood of the transformed (physical) parameters
        def loglike_single(u):
            return Model.likelihood(Model.prior(u))

        ## Model.prior returns a list of scalars; stack to a (Npar,) array
        def transform_single(u):
            return jnp.stack([jnp.asarray(theta_i) for theta_i in Model.prior(u)])

        if self.batch_size is None:
            loglike_batch = jax.vmap(loglike_single)
        else:
            ## chunked evaluation bounds device memory at batch_size likelihood evaluations
            loglike_batch = lambda U: jax.lax.map(loglike_single, U, batch_size=self.batch_size)

        self._dev = jax.devices(self.platform)[0]
        self._device_put = jax.device_put
        self._f_single = jax.jit(loglike_single)
        self._f_batch = jax.jit(loglike_batch)
        self._f_transform = jax.jit(jax.vmap(transform_single))

    def _put(self, x):
        '''Copy a host array to the target device as float64 (building the jitted functions if needed); jit then executes on that device.'''
        if self._dev is None:
            self._build()
        return self._device_put(np.asarray(x, dtype=np.float64), self._dev)

    def _bucket(self, n):
        '''Padded batch size for n points: next power of two, capped at max_batch.'''
        return min(_next_pow2(n), max(self.max_batch, n))

    def attach_batcher(self, batcher):
        '''Route calls made from batcher worker threads through the batcher.'''
        self._batcher = batcher

    def __call__(self, u):
        '''
        Log likelihood of one unit-cube point. Inside a BatchedLikelihoodPool worker the call is coalesced
        with those of the other workers; otherwise it is evaluated directly.

        Arguments
        -------------
        u (array) : (Npar,) point in the unit cube

        Returns
        -------------
        loglike (float)
        '''
        if self._batcher is not None and self._batcher.in_worker():
            return self._batcher.submit(u)
        return self.single(u)

    def single(self, u):
        '''Evaluate one point directly on the device (one host -> device and one device -> host transfer).'''
        ## transfer before looking up the jitted function, since _put builds it on first use
        u_dev = self._put(u)
        return float(self._f_single(u_dev))

    def batch(self, U):
        '''
        Evaluate many unit-cube points with batched device calls.

        Points are split into chunks of max_batch, and each chunk is padded (with copies of its first row) up to
        a power of two so that at most log2(max_batch)+2 batch shapes are ever compiled.

        Arguments
        -------------
        U (array) : (n, Npar) points in the unit cube

        Returns
        -------------
        loglike (array) : (n,) log likelihoods
        '''
        U = np.atleast_2d(np.asarray(U, dtype=np.float64))
        out = np.empty(len(U))
        for start in range(0, len(U), self.max_batch):
            chunk = U[start:start+self.max_batch]
            n = len(chunk)
            ## pad with a valid point rather than zeros to avoid spurious non-finite values
            pad = self._bucket(n) - n
            if pad > 0:
                chunk = np.concatenate([chunk, np.repeat(chunk[:1], pad, axis=0)])
            chunk_dev = self._put(chunk)
            out[start:start+n] = np.asarray(self._f_batch(chunk_dev))[:n]
        return out

    def transform(self, U):
        '''
        Map unit-cube samples to physical parameters with the Model prior.

        Arguments
        -------------
        U (array) : (n, Npar) points in the unit cube

        Returns
        -------------
        theta (array) : (n, Npar) physical parameters
        '''
        U_dev = self._put(np.atleast_2d(U))
        return np.asarray(self._f_transform(U_dev))

    def warmup(self):
        '''
        Compile the single-point function and every padded batch shape up front so that compilation never
        stalls a sampling run.

        Returns
        -------------
        compile_times (dict) : wall time in seconds of the first call for each batch shape (0 = single point)
        '''
        u0 = np.full(self.Npar, 0.5)
        compile_times = {}
        t0 = time.time()
        self.single(u0)
        compile_times[0] = time.time() - t0
        shapes = sorted({self._bucket(n) for n in range(1, self.max_batch+1)})
        for m in shapes:
            t0 = time.time()
            self.batch(np.tile(u0, (m, 1)))
            compile_times[m] = time.time() - t0
        return compile_times


class _Request():
    '''A pending likelihood evaluation from one worker thread.'''
    __slots__ = ('u', 'value', 'error', 'done')

    def __init__(self, u):
        self.u = u
        self.value = None
        self.error = None
        self.done = False


class BatchedLikelihoodPool():

    '''
    dynesty-compatible pool (provides .map and .size) that evaluates proposals in worker threads and coalesces their
    likelihood calls into batched device calls.

    Batching rule (combining barrier): a batch is evaluated as soon as every task that is still running is blocked
    on a likelihood request. The batch is therefore always the k-th request of each live task, independent of thread
    timing, so results are reproducible. No timeouts or dispatcher thread are needed.

    Arguments
    -------------
    loglike (JaxLikelihood) : likelihood to batch; its calls from this pool's workers are routed here
    size (int)              : number of worker threads; should equal the sampler's queue_size
    '''

    def __init__(self, loglike, size):
        self.loglike = loglike
        self.size = int(size)
        self._executor = ThreadPoolExecutor(max_workers=self.size, thread_name_prefix='blip-batch')
        self._cond = threading.Condition()
        self._n_active = 0
        self._pending = []
        self._local = threading.local()
        ## diagnostics
        self.n_batches = 0
        self.n_evals = 0
        loglike.attach_batcher(self)

    def in_worker(self):
        '''True if the calling thread is currently running a task for this pool.'''
        return getattr(self._local, 'active', False)

    def map(self, func, iterable):
        '''Apply func to every element of iterable using the worker threads, preserving order.'''
        args = list(iterable)
        results = []
        ## at most `size` tasks at once, so every counted task has a thread and the barrier can't deadlock
        for start in range(0, len(args), self.size):
            chunk = args[start:start+self.size]
            with self._cond:
                self._n_active = len(chunk)
            futures = [self._executor.submit(self._run, func, a) for a in chunk]
            ## wait for all before collecting, so an exception can't leave tasks running into the next chunk
            wait(futures)
            results.extend(f.result() for f in futures)
        return results

    def _run(self, func, arg):
        '''Run one task in a worker thread and update the active-task count when it finishes.'''
        self._local.active = True
        try:
            return func(arg)
        finally:
            self._local.active = False
            with self._cond:
                self._n_active -= 1
                ## this task finishing may leave all remaining tasks blocked on requests
                self._maybe_flush()

    def submit(self, u):
        '''Queue one likelihood request from a worker thread and block until its batch has been evaluated.'''
        req = _Request(np.asarray(u, dtype=np.float64))
        with self._cond:
            self._pending.append(req)
            self._maybe_flush()
            while not req.done:
                self._cond.wait()
        if req.error is not None:
            raise req.error
        return req.value

    def _maybe_flush(self):
        '''Evaluate the pending batch if every active task is waiting on it. Must hold self._cond.'''
        if not self._pending or len(self._pending) < self._n_active:
            return
        batch, self._pending = self._pending, []
        ## the lock stays held during the device call; by construction every other active task is waiting anyway
        try:
            vals = self.loglike.batch(np.stack([req.u for req in batch]))
            for req, val in zip(batch, vals):
                req.value = float(val)
        except BaseException as err:
            for req in batch:
                req.error = err
        for req in batch:
            req.done = True
        self.n_batches += 1
        self.n_evals += len(batch)
        self._cond.notify_all()

    def close(self):
        '''Shut down the worker threads.'''
        self._executor.shutdown(wait=True)

    def join(self):
        '''No-op; present for multiprocessing.Pool API compatibility.'''
        return


def _attach_pool(engine, pool):
    '''
    Attach a pool to a dynesty sampler, or detach it with pool=None.

    dynesty >= 3.0 maps through engine.mapper; dynesty 2.x through engine.M and engine.loglikelihood.pool.
    '''
    mapper = pool.map if pool is not None else map
    engine.pool = pool
    if hasattr(engine, 'mapper'):
        engine.mapper = mapper
    else:
        engine.M = mapper
        engine.loglikelihood.pool = pool


###################################################
###              DYNESTY ENGINE                 ###
###################################################

class dynesty_engine():

    '''
    Class for interfacing with dynesty sampler. This method also contains the
    priors definition for all models written to work with the dynesty sampler.

    The BLIP prior and likelihood are evaluated as one jitted JAX function of the unit cube (JaxLikelihood) on the
    CPU or GPU, so dynesty is given an identity prior transform and its stored samples are unit-cube values.
    Physical posterior samples are recovered with the Model prior after sampling.
    '''


    @classmethod
    def define_engine(cls, lisaobj, params, nlive, nthread, randst, gpu=False):
        '''
        Build the dynesty sampler around a jitted BLIP likelihood.

        Arguments
        -------------
        lisaobj (LISA)      : LISA object holding the analysis Model
        params (dict)       : run parameters (uses sample_method and gpu_batch_size)
        nlive (int)         : number of live points
        nthread (int)       : number of parallel proposals (dynesty queue_size); if > 1, their likelihood calls
                              are batched into single device calls by a BatchedLikelihoodPool
        randst (Generator)  : numpy random state
        gpu (bool)          : evaluate the likelihood on the GPU if True, else on the CPU

        Returns
        -------------
        engine (NestedSampler), parameters (dict)
        '''
        Model = lisaobj.Model
        loglike = cls.build_likelihood(Model, params, nthread, gpu)

        # parallel proposals share one batched likelihood (threads, not processes, so this is CUDA-safe)
        if nthread > 1:
            pool = BatchedLikelihoodPool(loglike, nthread)
            queue_size = nthread
        else:
            pool = None
            queue_size = None

        # blm phases are periodic in the unit cube
        periodic = list(Model.blm_phase_idx) if len(Model.blm_phase_idx) > 0 else None

        # only proposals and likelihoods use the pool; the identity prior transform and bound updates stay serial
        engine = NestedSampler(loglike, unit_cube_transform, Model.Npar,\
                    bound='multi', sample=params['sample_method'], nlive=nlive, pool=pool, queue_size=queue_size, rstate = randst,\
                    periodic=periodic, use_pool={'prior_transform':False,'update_bound':False})
        ## keep a direct handle on the likelihood (pickled with the engine) for resuming and post-processing
        engine.blip_loglike = loglike


        # print npar
        print("Npar = " + str(Model.Npar))

        return engine, Model.parameters

    @staticmethod
    def build_likelihood(Model, params, nthread, gpu):
        '''
        Create and pre-compile the JaxLikelihood for this run.

        Arguments
        -------------
        Model (Model)   : BLIP analysis Model
        params (dict)   : run parameters (uses gpu_batch_size)
        nthread (int)   : largest batch the sampler will request
        gpu (bool)      : target the GPU if True, else the CPU

        Returns
        -------------
        loglike (JaxLikelihood)
        '''
        platform = 'gpu' if gpu else 'cpu'
        loglike = JaxLikelihood(Model, platform=platform, max_batch=nthread, batch_size=params.get('gpu_batch_size'))
        print("Compiling likelihood on {} (batch shapes up to {})...".format(platform, max(nthread,1)))
        compile_times = loglike.warmup()
        print("Likelihood compiled in {:.1f} s.".format(sum(compile_times.values())))
        return loglike

    def load_engine(params,randst,nthread,gpu=False):
        ## load engine from previous checkpoint
        ## nqueue is set to a negative number to trigger the queue to be refilled before the first iteration.
        ## randomstate cannot be saved, so we need to set that as well
        resume_file = params['out_dir']+'/checkpoint.pickle'
        if os.path.isfile(resume_file):
            print("Loading interrupted analysis from last checkpoint...")
            with open(resume_file,'rb') as file:
                engine, parameters = dill.load(file)
                if engine.added_live:
                    engine._remove_live_points()
                engine.nqueue = -1
                engine.rstate = randst
                ## the compiled likelihood is not saved; retarget and recompile it for this session
                loglike = engine.blip_loglike
                loglike.platform = 'gpu' if gpu else 'cpu'
                loglike.max_batch = max(nthread,1)
                loglike.batch_size = params.get('gpu_batch_size')
                loglike.warmup()
                ## threads hold no pickled state, so batched proposals can be resumed (unlike process pools)
                pool = BatchedLikelihoodPool(loglike, nthread) if nthread > 1 else None
                _attach_pool(engine, pool)
                engine.queue_size = max(nthread,1)
        else:
            raise TypeError("Checkpoint file <{}> does not exist. Cannot resume from checkpoint.".format(resume_file))

        return engine, parameters

    @staticmethod
    def get_posterior(engine):
        '''
        Equal-weight physical posterior samples and evidence from a finished run.

        Returns
        -------------
        post_samples (array) : (Nsamples, Npar) physical parameters
        logz, logzerr (array) : evidence and its error along the run
        '''
        # re-scale weights to have a maximum of one
        res = engine.results
        weights = np.exp(res['logwt'] - res['logz'][-1])
        weights[-1] = 1 - np.sum(weights[0:-1])

        # dynesty's samples are unit-cube values (identity prior transform); map them to physical parameters
        post_samples = engine.blip_loglike.transform(resample_equal(res.samples, weights))

        # Pull the evidence and the evidence error
        logz = res['logz']
        logzerr = res['logzerr']

        return post_samples, logz, logzerr

    @staticmethod
    def run_engine_with_checkpointing(engine,parameters,interval,checkpoint_file,step=1000):

       # -------------------- Run nested sampler ---------------------------
        pool = engine.pool
        old_ncall = engine.ncall
        start_time = time.time()
        while True:
            engine.run_nested(dlogz=0.5,maxiter=step,print_progress=True)
            if engine.ncall == old_ncall:
                break
            old_ncall = engine.ncall
            if os.path.isfile(checkpoint_file):
                last_checkpoint_s = time.time() - os.path.getmtime(checkpoint_file)
            else:
                last_checkpoint_s = time.time() - start_time
            if last_checkpoint_s > interval:
                print("Checkpointing...")
                ## pause the pool for saving
                _attach_pool(engine, None)
                ## save (JaxLikelihood drops its compiled functions and pool reference when pickled)
                if dill.pickles([engine,parameters]):
                    temp_file = checkpoint_file + ".temp"
                    with open(temp_file, "wb") as file:
                        dill.dump([engine,parameters], file)
                    shutil.move(temp_file, checkpoint_file)
                else:
                    print("Warning: Cannot write checkpoint file, job cannot resume if interrupted.")
                ## restart pool if needed
                _attach_pool(engine, pool)
                ## removes live points lumped in with dead points when previous sampling call concluded
                ## unsure why dynesty requires this, but it seems to be deeply important
                if engine.added_live:
                    engine._remove_live_points()

        return dynesty_engine.get_posterior(engine)

    @staticmethod
    def run_engine(engine):


       # -------------------- Run nested sampler ---------------------------
        engine.run_nested(dlogz=0.5,print_progress=True )

        return dynesty_engine.get_posterior(engine)




