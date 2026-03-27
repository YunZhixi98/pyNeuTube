from __future__ import annotations

"""
Deprecated local wheel builder.

This helper is kept only for ad-hoc local builds with temporary conda environments.
For release-quality wheels, especially Linux manylinux wheels intended for PyPI,
prefer `cibuildwheel` instead.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIST_DIR = REPO_ROOT / "dist"
DEFAULT_WHEELHOUSE = REPO_ROOT / ".wheelhouse"
ENV_PREFIX = "pyneutube-wheel"
CONDA_CMD: list[str] | None = None


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("==>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def resolve_conda_command() -> list[str]:
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        return [conda_exe]

    for candidate in ("conda", "conda.exe", "conda.bat"):
        resolved = shutil.which(candidate)
        if resolved:
            return [resolved]

    condabin = Path(sys.base_prefix).parent / "condabin"
    for candidate in ("conda.bat", "conda.exe"):
        resolved = condabin / candidate
        if resolved.exists():
            return [str(resolved)]

    raise RuntimeError("`conda` was not found. Activate a conda shell or set CONDA_EXE.")


def require_conda() -> None:
    global CONDA_CMD
    CONDA_CMD = resolve_conda_command()


def env_name(version: str) -> str:
    return f"{ENV_PREFIX}-py{version.replace('.', '')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build wheels for multiple Python versions using temporary conda environments."
    )
    parser.add_argument(
        "--python",
        nargs="+",
        default=["3.10", "3.11", "3.12"],
        dest="python_versions",
        help="Python versions to build against.",
    )
    parser.add_argument(
        "--wheelhouse",
        default=str(DEFAULT_WHEELHOUSE),
        help="Directory where built wheels are collected.",
    )
    parser.add_argument(
        "--build-sdist",
        action="store_true",
        help="Also build one source distribution using the first Python version.",
    )
    return parser.parse_args()


def conda_env_exists(name: str) -> bool:
    result = subprocess.run(
        [*CONDA_CMD, "env", "list"],
        check=True,
        capture_output=True,
        text=True,
    )
    return any(line.split() and line.split()[0] == name for line in result.stdout.splitlines())


def create_env(name: str, python_version: str) -> None:
    if conda_env_exists(name):
        raise RuntimeError(f"Conda environment already exists: {name}")
    run([*CONDA_CMD, "create", "-y", "-n", name, f"python={python_version}"])


def conda_run(name: str, *args: str) -> None:
    run([*CONDA_CMD, "run", "--no-capture-output", "-n", name, *args], cwd=REPO_ROOT)


def clean_build_state() -> None:
    shutil.rmtree(REPO_ROOT / "build", ignore_errors=True)


def collect_artifacts(target_dir: Path) -> None:
    if not BUILD_DIST_DIR.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for artifact in BUILD_DIST_DIR.iterdir():
        if artifact.is_file():
            shutil.move(str(artifact), str(target_dir / artifact.name))
    shutil.rmtree(BUILD_DIST_DIR, ignore_errors=True)


def remove_env(name: str) -> None:
    if conda_env_exists(name):
        try:
            run([*CONDA_CMD, "env", "remove", "-y", "-n", name])
        except subprocess.CalledProcessError as exc:
            print(f"Warning: failed to remove conda env {name}: {exc}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    require_conda()

    print("WARNING: tools/release/build_wheels_conda.py is deprecated.")
    print("WARNING: Prefer `python -m cibuildwheel --output-dir wheelhouse` for release builds.")

    wheelhouse = Path(args.wheelhouse).resolve()
    wheelhouse.mkdir(parents=True, exist_ok=True)

    created_envs: list[str] = []
    try:
        for index, python_version in enumerate(args.python_versions):
            name = env_name(python_version)
            print(f"==> Creating temporary conda env: {name} (python={python_version})")
            create_env(name, python_version)
            created_envs.append(name)

            print(f"==> Installing build tools into {name}")
            conda_run(name, "python", "-m", "pip", "install", "--upgrade", "pip")
            conda_run(
                name,
                "python",
                "-m",
                "pip",
                "install",
                "build",
                "setuptools",
                "wheel",
                "numpy",
                "Cython",
            )

            clean_build_state()

            print(f"==> Building wheel for Python {python_version}")
            conda_run(name, "python", "-m", "build", "--wheel")
            if args.build_sdist and index == 0:
                print("==> Building sdist")
                conda_run(name, "python", "-m", "build", "--sdist")

            print("==> Collecting artifacts")
            collect_artifacts(wheelhouse)
    finally:
        clean_build_state()
        for name in created_envs:
            print(f"==> Removing temporary conda env: {name}")
            remove_env(name)

    print(f"==> Done. Built artifacts are in: {wheelhouse}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
