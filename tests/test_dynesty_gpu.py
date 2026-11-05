import threading
from pathlib import Path
from types import SimpleNamespace

import dill
import jax
import numpy as np
import pytest

import blip
from blip.config import parse_config
from blip.run_blip import LISA
from blip.src.dynesty_engine import dynesty_engine, JaxLikelihood, BatchedLikelihoodPool, _attach_pool

blip_root_dir = Path(blip.__file__).resolve().parent.parent

# The 3x3 XYZ model covariances in params_test_cpu.ini have condition numbers up to ~1e14, so any two float64
# evaluation orders (eager vs jit, vmap vs single) differ at the ~1e-5 relative level. This is not a bug in the wrapper.
RTOL = 1e-4


def _gpu_available():
    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False


PLATFORMS = ["cpu"] + (["gpu"] if _gpu_available() else [])


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """BLIP analysis Model for params_test_cpu.ini (noise + powerlaw_isgwb), built once per module."""
    params, inj, misc = parse_config(blip_root_dir / "params_test_cpu.ini", resume=False)
    params["out_dir"] = str(tmp_path_factory.mktemp("run_result"))
    return LISA(params, inj).Model


@pytest.fixture(scope="module")
def unit_points(model):
    """Random unit-cube points."""
    return np.random.default_rng(0).random((13, model.Npar))


@pytest.mark.parametrize("platform", PLATFORMS)
def test_single_matches_eager(model, unit_points, platform):
    loglike = JaxLikelihood(model, platform=platform)
    eager = np.array([float(model.likelihood(model.prior(u))) for u in unit_points])
    jitted = np.array([loglike(u) for u in unit_points])
    np.testing.assert_allclose(jitted, eager, rtol=RTOL)


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("n", [1, 5, 8, 13])
def test_batch_matches_single(model, unit_points, platform, n):
    # n=5 exercises padding, n=13 exercises chunking (max_batch=8) plus padding
    loglike = JaxLikelihood(model, platform=platform, max_batch=8)
    single = np.array([loglike.single(u) for u in unit_points[:n]])
    np.testing.assert_allclose(loglike.batch(unit_points[:n]), single, rtol=RTOL)


def test_chunked_batch_matches_vmap(model, unit_points):
    full = JaxLikelihood(model, platform="cpu", max_batch=8)
    chunked = JaxLikelihood(model, platform="cpu", max_batch=8, batch_size=2)
    np.testing.assert_allclose(chunked.batch(unit_points), full.batch(unit_points), rtol=RTOL)


def test_transform_matches_prior(model, unit_points):
    loglike = JaxLikelihood(model, platform="cpu")
    expected = np.array([[float(theta_i) for theta_i in model.prior(u)] for u in unit_points])
    np.testing.assert_allclose(loglike.transform(unit_points), expected, rtol=1e-12)


def test_warmup_compiles_all_buckets(model):
    loglike = JaxLikelihood(model, platform="cpu", max_batch=12)
    # single point (0) plus padded shapes 1, 2, 4, 8 and the cap 12
    assert sorted(loglike.warmup()) == [0, 1, 2, 4, 8, 12]


def test_pickle_drops_compiled_state(model, unit_points):
    loglike = JaxLikelihood(model, platform="cpu", max_batch=4)
    pool = BatchedLikelihoodPool(loglike, 4)
    before = loglike(unit_points[0])
    restored = dill.loads(dill.dumps(loglike))
    assert restored._f_single is None and restored._batcher is None
    assert restored(unit_points[0]) == before
    pool.close()


def test_pool_lockstep_batching(model):
    loglike = JaxLikelihood(model, platform="cpu", max_batch=4)
    pool = BatchedLikelihoodPool(loglike, 4)

    def task(arg):
        """Evaluate k random points sequentially, like one dynesty proposal chain."""
        seed, k = arg
        rng = np.random.default_rng(seed)
        return [loglike(rng.random(model.Npar)) for _ in range(k)]

    # ragged chains: batches shrink as chains finish
    args = [(0, 1), (1, 3), (2, 2), (3, 5)]
    batched = pool.map(task, args)
    pool.close()
    loglike._batcher = None
    serial = [task(a) for a in args]
    for b, s in zip(batched, serial):
        np.testing.assert_allclose(b, s, rtol=RTOL)
    # one batch per step of the longest chain, one evaluation per call
    assert pool.n_batches == 5
    assert pool.n_evals == 11


def test_pool_error_does_not_deadlock(model):
    loglike = JaxLikelihood(model, platform="cpu", max_batch=4)
    pool = BatchedLikelihoodPool(loglike, 4)

    def task(i):
        """Task 0 fails after one call; the others keep requesting likelihoods."""
        loglike(np.full(model.Npar, 0.5))
        if i == 0:
            raise RuntimeError("boom")
        return [loglike(np.full(model.Npar, 0.4)) for _ in range(3)]

    outcome = {}

    def run():
        try:
            pool.map(task, range(4))
        except RuntimeError as err:
            outcome["error"] = err

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=120)
    assert not thread.is_alive(), "pool deadlocked after a task raised"
    assert "error" in outcome
    pool.close()


def _short_run(model, nthread, seed, maxiter=150):
    """Build a dynesty engine through dynesty_engine and run a fixed number of iterations."""
    params = {"sample_method": "rslice", "gpu_batch_size": None}
    engine, _ = dynesty_engine.define_engine(SimpleNamespace(Model=model), params, 50, nthread,
                                             np.random.default_rng(seed))
    engine.run_nested(maxiter=maxiter, print_progress=False)
    return engine


def _mapper(engine):
    """The map function dynesty will use for proposals (engine.mapper in dynesty >= 3, engine.M in 2.x)."""
    return engine.mapper if hasattr(engine, "mapper") else engine.M


def test_attach_pool_detach_and_reattach(model):
    engine = _short_run(model, nthread=4, seed=4, maxiter=10)
    pool = engine.pool
    assert _mapper(engine) == pool.map
    _attach_pool(engine, None)
    assert engine.pool is None and _mapper(engine) is map
    _attach_pool(engine, pool)
    assert engine.pool is pool and _mapper(engine) == pool.map
    pool.close()


def test_dynesty_batched_run_is_reproducible(model):
    run1 = _short_run(model, nthread=4, seed=1)
    run2 = _short_run(model, nthread=4, seed=1)
    np.testing.assert_array_equal(run1.results.logl, run2.results.logl)
    post_samples, logz, logzerr = dynesty_engine.get_posterior(run1)
    assert post_samples.shape[1] == model.Npar
    assert np.all(np.isfinite(post_samples))
    run1.pool.close()
    run2.pool.close()


def test_dynesty_checkpoint_resume(model, tmp_path):
    engine = _short_run(model, nthread=4, seed=2, maxiter=100)
    engine.pool.close()
    with open(tmp_path / "checkpoint.pickle", "wb") as file:
        dill.dump([engine, model.parameters], file)
    params = {"out_dir": str(tmp_path), "gpu_batch_size": None}
    resumed, _ = dynesty_engine.load_engine(params, np.random.default_rng(3), 4)
    ncall_before = resumed.ncall
    resumed.run_nested(maxiter=50, print_progress=False)
    assert resumed.ncall > ncall_before
    # proposals must go through the re-attached pool (on dynesty 3.x a stale engine.M would silently run serially)
    assert resumed.pool.n_batches > 0
    resumed.pool.close()
