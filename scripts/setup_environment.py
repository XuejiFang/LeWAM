"""Install the RoboTwin evaluation environment."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CUROBO_REVISION = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"


def run(*args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Install uv before running this script.")
    environment = ROOT / ".venvs" / "robotwin"
    project = ROOT / "environments/robotwin"
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(environment)}
    run(uv, "sync", "--project", project, "--locked", env=env)
    python = environment / "bin/python"
    curobo = ROOT / ".cache/curobo"
    if not curobo.exists():
        run("git", "clone", "--depth", "1", "--branch", "v0.7.8",
            "https://github.com/NVlabs/curobo.git", curobo)
    revision = subprocess.check_output(
        ["git", "-C", str(curobo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != CUROBO_REVISION:
        raise RuntimeError(f"Expected CuRobo {CUROBO_REVISION}, found {revision} in {curobo}")

    # Compile against the environment's installed PyTorch, as required by CuRobo.
    run(uv, "pip", "install", "--python", python, "--no-build-isolation",
        "--no-deps", curobo, env=env)

    # RoboTwin's official installation removes this collision check in MPLib.
    planner = environment / "lib/python3.10/site-packages/mplib/planner.py"
    original = "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:"
    patched = "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"
    source = planner.read_text()
    if source.count(original) == 1:
        planner.write_text(source.replace(original, patched))
    elif source.count(patched) != 1:
        raise RuntimeError(f"Unexpected MPLib planner source: {planner}")


if __name__ == "__main__":
    main()
