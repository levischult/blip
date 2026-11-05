## Additions for tests/test_end_to_end.py once the dynesty GPU changes are integrated into BLIP.
import os
import shutil
from pathlib import Path

import jax
import numpy as np
import pytest

import blip
from blip.run_blip import run_pipeline
from blip.config import parse_config

blip_root_dir = Path(blip.__file__).resolve().parent.parent
## params_test_dynesty_gpu.ini sits next to the tests directory (repo root after integration, CLD/filesforblip before)
dynesty_gpu_params = Path(__file__).resolve().parent.parent / "params_test_dynesty_gpu.ini"

def _gpu_available():
    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False

# This runs BLIP's dynesty sampler with a batched GPU likelihood and fails if any exception is raised.
@pytest.mark.skipif(not _gpu_available(), reason="params_test_dynesty_gpu.ini requests a GPU (N_GPU=1), but JAX has no GPU backend")
def test_end_to_end_dynesty_gpu(tmp_path):
    target = tmp_path.absolute() / "params.ini"
    shutil.copy(dynesty_gpu_params, target)

    # change output directory to be inside tmp_path
    params, inj, misc = parse_config(target, resume=False)
    params["out_dir"] = str(tmp_path / "run_result")
    config = (params, inj, misc)

    run_pipeline(config, resume=False)
    assert np.loadtxt(tmp_path / "run_result" / "post_samples.txt").ndim == 2


# Same as test_end_to_end_cpu, but with 4 parallel proposals batched on the CPU.
def test_end_to_end_dynesty_cpu_batched(tmp_path):
    source = blip_root_dir / "params_test_cpu.ini"
    target = tmp_path.absolute() / "params.ini"
    shutil.copy(source, target)

    # change output directory to be inside tmp_path
    params, inj, misc = parse_config(target, resume=False)
    params["out_dir"] = str(tmp_path / "run_result")
    misc["nthread"] = 4
    config = (params, inj, misc)

    run_pipeline(config, resume=False)
    assert np.loadtxt(tmp_path / "run_result" / "post_samples.txt").ndim == 2
