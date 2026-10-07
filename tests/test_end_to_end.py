import os
import shutil
from pathlib import Path

import jax
import pytest

from blip.run_blip import run_pipeline
from blip.config import parse_config

blip_root_dir = Path(os.path.dirname(os.path.dirname(__file__))).absolute()

def _gpu_available():
    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False

# This runs BLIP on params_test.ini and fails if any exception is raised.
@pytest.mark.skipif(not _gpu_available(), reason="params_test.ini requests a GPU (N_GPU=1), but JAX has no GPU backend")
def test_end_to_end(tmp_path):
    source = blip_root_dir / "params_test.ini"
    target = tmp_path.absolute() / "params.ini"
    shutil.copy(source, target)

    # change output directory to be inside tmp_path
    params, inj, misc = parse_config(target, resume=False)
    params["out_dir"] = str(tmp_path / "run_result")
    config = (params, inj, misc)

    run_pipeline(config, resume=False)


def test_end_to_end_cpu(tmp_path):
    source = blip_root_dir / "params_test_cpu.ini"
    target = tmp_path.absolute() / "params.ini"
    shutil.copy(source, target)

    # change output directory to be inside tmp_path
    params, inj, misc = parse_config(target, resume=False)
    params["out_dir"] = str(tmp_path / "run_result")
    config = (params, inj, misc)

    run_pipeline(config, resume=False)
