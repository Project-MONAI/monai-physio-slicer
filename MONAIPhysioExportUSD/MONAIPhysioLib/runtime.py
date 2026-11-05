# SPDX-License-Identifier: Apache-2.0
"""Select Slicer's interpreter and protect bundled packages during setup."""

from __future__ import annotations

import importlib.metadata
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional


MINIMUM_MONAI_PHYSIO = "2026.11.1"
REQUIREMENT = f"monai-physio>={MINIMUM_MONAI_PHYSIO}"

# PyTorch builds published on download.pytorch.org for CPython 3.12 (Windows
# and Linux): version -> (matching torchvision, computation backends). CUDA
# 12.8+ builds are needed for RTX 50 / Blackwell GPUs; cu128 stops at 2.11.
TORCH_BUILDS = {
    "2.10.0": ("0.25.0", ("cu126", "cu128", "cu130", "cpu")),
    "2.11.0": ("0.26.0", ("cu126", "cu128", "cu130", "cpu")),
    "2.12.1": ("0.27.1", ("cu126", "cu130", "cu132", "cpu")),
    "2.13.0": ("0.28.0", ("cu126", "cu130", "cu132", "cpu")),
    "2.14.1": ("0.29.1", ("cu126", "cu130", "cu132", "cpu")),
}
# Builds with a prebuilt torch-scatter (cp312) on data.pyg.org, which names
# its pages by minor version. Others need torch-scatter built from source.
TORCH_SCATTER_WHEELS = {
    "2.10": ("cu126", "cu128", "cpu"),
    "2.11": ("cu126", "cu128", "cpu"),
    "2.12": ("cu126", "cpu"),
}
# Assets of this extension's GitHub release, shared with the sample data.
RELEASE_URL = (
    "https://github.com/Project-MONAI/monai-physio-slicer/releases/download/Data"
)
# torch-scatter 2.1.2 wheels built for this extension where data.pyg.org has
# none (CPython 3.12, Windows x64 only), named like PyG's: 2.1.2+pt214cu132.
RELEASE_TORCH_SCATTER_WHEELS = {
    "2.12": ("cu130", "cu132"),
    "2.13": ("cu130", "cu132", "cpu"),
    "2.14": ("cu130", "cu132", "cpu"),
}


def monai_physio_problem() -> Optional[str]:
    """Why the installed MONAI-Physio cannot run the workers, or None."""
    from packaging.version import Version

    try:
        version = importlib.metadata.version("monai-physio")
    except importlib.metadata.PackageNotFoundError:
        return (
            "MONAI-Physio is not installed. Click 'Set up MONAI-Physio in "
            "Slicer...' first."
        )
    if Version(version) < Version(MINIMUM_MONAI_PHYSIO):
        return (
            f"MONAI-Physio {version} is installed; these modules need "
            f"{MINIMUM_MONAI_PHYSIO} or newer. Click 'Set up MONAI-Physio in "
            "Slicer...' to update it."
        )
    return None


def installed_torch() -> Optional[str]:
    """Installed PyTorch version with its build label, e.g. 2.10.0+cu126."""
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_choices() -> list[tuple[str, str]]:
    """(label, build) pairs for the PyTorch menu, newest first."""
    choices = []
    for version in sorted(TORCH_BUILDS, key=lambda v: tuple(map(int, v.split("."))),
                          reverse=True):
        for backend in TORCH_BUILDS[version][1]:
            if backend == "cpu":
                label = f"{version} CPU only"
            else:
                cuda = f"{backend[2:-1]}.{backend[-1]}"
                label = f"{version} CUDA {cuda}"
                if cuda in ("12.8", "13.0", "13.2"):
                    label += ", RTX 50 / Blackwell"
            if not has_torch_scatter_wheel(f"{version}+{backend}"):
                label += ", no prebuilt torch-scatter"
            choices.append((label, f"{version}+{backend}"))
    return choices


def _pyg_page(build: str) -> tuple[str, str]:
    """(minor version, backend) naming the data.pyg.org page for a build."""
    version, _, backend = build.partition("+")
    return ".".join(version.split(".")[:2]), backend


def release_torch_scatter_wheel(build: str) -> Optional[str]:
    """URL of this extension's torch-scatter wheel for a build, if one exists.

    Installed by exact URL, never through a find-links listing, so pip
    cannot pick a wheel built for another PyTorch.
    """
    minor, backend = _pyg_page(build)
    if (
        sys.platform != "win32"
        or sys.version_info[:2] != (3, 12)
        or backend not in RELEASE_TORCH_SCATTER_WHEELS.get(minor, ())
    ):
        return None
    tag = f"pt{minor.replace('.', '')}{backend}"
    return f"{RELEASE_URL}/torch_scatter-2.1.2+{tag}-cp312-cp312-win_amd64.whl"


def has_torch_scatter_wheel(build: str) -> bool:
    minor, backend = _pyg_page(build)
    return (
        backend in TORCH_SCATTER_WHEELS.get(minor, ())
        or release_torch_scatter_wheel(build) is not None
    )


def install_torch_scatter(
    build: str, force: bool = False, log: Callable[[str], Any] = print
) -> bool:
    """Install torch-scatter for a PyTorch build from its prebuilt wheel.

    Uses PyTorch Geometric's wheel index, or else this extension's release
    asset. pip would otherwise compile it from source, which needs a
    compiler and a CUDA toolkit matching PyTorch. Returns False when no
    wheel exists.
    """
    import slicer

    minor, backend = _pyg_page(build)
    flags = ["--force-reinstall", "--no-cache-dir", "--no-deps"] if force else []
    release = release_torch_scatter_wheel(build)
    try:
        if backend in TORCH_SCATTER_WHEELS.get(minor, ()) or release is None:
            slicer.util.pip_install([
                "torch-scatter", "--only-binary", "torch-scatter",
                "-f", f"https://data.pyg.org/whl/torch-{minor}.0+{backend}.html",
                *flags,
            ])
        else:
            log(f"Installing torch-scatter for PyTorch {build} from {release}")
            slicer.util.pip_install([release, *(flags or ["--no-deps"])])
    except Exception:
        log(
            f"No prebuilt torch-scatter for PyTorch {build}: MONAI-Physio's "
            "motion network (PhysicsNeMo) needs torch-scatter built for this "
            "PyTorch (a compiler and matching CUDA toolkit)."
        )
        return False
    log(f"Installed torch-scatter for PyTorch {build}.")
    return True


def _torch_install_arguments(build: str) -> tuple[str, str, str]:
    version, _, backend = build.partition("+")
    if version not in TORCH_BUILDS or backend not in TORCH_BUILDS[version][1]:
        raise ValueError(f"Unknown PyTorch build: {build}")
    return version, TORCH_BUILDS[version][0], backend


def install_torch(build: str, log: Callable[[str], Any] = print) -> None:
    """Replace Slicer's PyTorch with the given build, e.g. 2.11.0+cu128.

    Uses the PyTorch extension's light-the-torch installer when the extension
    is installed, otherwise pip with the PyTorch wheel index. --force-reinstall
    and --no-cache-dir make a backend-only change (cu126 -> cu128 of the same
    version) actually happen; --no-deps keeps that forced step from
    reinstalling Slicer's numpy, and a normal second pass then adds any new
    dependencies. torch-scatter is compiled against one PyTorch, so it is
    replaced from PyTorch Geometric's prebuilt wheels when one exists.
    """
    import slicer

    version, torchvision, backend = _torch_install_arguments(build)
    if "torch" in sys.modules:
        raise RuntimeError(
            "PyTorch is already loaded in this Slicer session, so its files "
            "cannot be replaced. Restart Slicer and change PyTorch before "
            "using modules that load it."
        )
    packages = [f"torch=={version}", f"torchvision=={torchvision}"]
    forced = ["--force-reinstall", "--no-cache-dir", "--no-deps"]
    try:
        import PyTorchUtils
    except ImportError:
        PyTorchUtils = None
    if PyTorchUtils is not None:
        log(f"Installing PyTorch {build} with the PyTorch extension...")
        logic = PyTorchUtils.PyTorchUtilsLogic()
        try:
            import light_the_torch._patch  # noqa: F401
        except ImportError:
            logic._installLightTheTorch()
        backend_arg = f"--pytorch-computation-backend={backend}"
        for extra in (forced, []):
            slicer.util._executePythonModule(
                "light_the_torch", ["install", *packages, backend_arg, *extra]
            )
    else:
        log(
            f"Installing PyTorch {build} with pip (install the PyTorch "
            "extension to have Slicer manage PyTorch builds)..."
        )
        index = ["--extra-index-url", f"https://download.pytorch.org/whl/{backend}"]
        local = [f"torch=={version}+{backend}", f"torchvision=={torchvision}+{backend}"]
        for extra in (forced, []):
            slicer.util.pip_install([*local, *index, *extra])
    install_torch_scatter(build, force=True, log=log)


def python_slicer() -> Path:
    """Locate Slicer's Python launcher without substituting another Python."""
    import slicer

    name = "PythonSlicer.exe" if sys.platform == "win32" else "PythonSlicer"
    candidates = [
        Path(slicer.app.slicerHome) / "bin" / name,
        Path(slicer.app.applicationDirPath()) / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "PythonSlicer was not found in this Slicer installation. "
        "Reinstall Slicer or use a standard Slicer build."
    )


def bundled_constraints() -> list[str]:
    """Constrain existing numerical packages; refuse ambiguous VTK installs.

    These constraints protect versions already supplied by the application,
    rather than imposing an extension-specific VTK/NumPy version. pip cannot
    safely resolve VTK if the bundled build has no distribution metadata.
    """
    if sys.version_info < (3, 11):
        raise RuntimeError(
            "MONAI-Physio requires Python 3.11 or newer. This Slicer uses "
            f"Python {sys.version_info.major}.{sys.version_info.minor}. "
            "Use a compatible Slicer build."
        )
    import numpy as np
    import vtk

    constraints = []
    for name in ("numpy", "vtk", "itk", "SimpleITK", "torch"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name in {"numpy", "vtk"}:
                raise RuntimeError(
                    f"Slicer's bundled {name} has no pip distribution metadata. "
                    "Automatic dependency installation cannot protect it. "
                    "This Slicer build is not supported."
                ) from None
            continue
        loaded_version = {
            "numpy": np.__version__,
            "vtk": vtk.vtkVersion.GetVTKVersion(),
        }.get(name)
        if loaded_version and loaded_version.split("+")[0] != version.split("+")[0]:
            raise RuntimeError(
                f"Loaded {name} ({loaded_version}) differs from pip metadata "
                f"({version}). Restart Slicer."
            )
        constraints.append(f"{name}=={version}")
    return constraints


def requirements_of(
    distribution: str, skip: tuple[str, ...] = (), unpin: tuple[str, ...] = ()
) -> list[str]:
    """Installed distribution's requirements that apply to this environment.

    Extras are left out. Names in skip are dropped; names in unpin keep only
    the project name, so protected packages are governed by constraints alone.
    """
    from packaging.requirements import Requirement

    importlib.invalidate_caches()
    result = []
    for text in importlib.metadata.requires(distribution) or []:
        requirement = Requirement(text)
        name = requirement.name.lower().replace("_", "-")
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        if name in skip:
            continue
        result.append(requirement.name if name in unpin else text.split(";")[0].strip())
    return result


def install_monai_physio(
    constraints: Path, install: Callable[[list[str]], Any]
) -> None:
    """Install MONAI-Physio while keeping antspyx from fighting Slicer's NumPy.

    antspyx declares numpy<2.4 and scipy<1.16, which cannot hold with a
    Slicer whose NumPy is newer; pip then backtracks into scipy source builds
    and fails. antspyx runs on the newer NumPy, so it is installed without
    dependency resolution and its remaining requirements are added directly.
    """
    base = ["-c", str(constraints)]
    install([*base, REQUIREMENT, "--no-deps"])
    others = requirements_of("monai-physio", skip=("antspyx",))
    install([*base, *others])
    ants = next(
        (r for r in importlib.metadata.requires("monai-physio") or []
         if r.lower().startswith("antspyx")),
        "antspyx",
    ).split(";")[0].strip()
    install([*base, ants, "--no-deps"])
    install([*base, *requirements_of("antspyx", unpin=("numpy", "scipy"))])


def ensure_native_dependencies(
    torch_build: Optional[str] = None, log: Callable[[str], Any] = print
) -> bool:
    """Offer Slicer's standard dependency installation only on user action.

    torch_build, when it differs from the installed PyTorch, replaces PyTorch
    first (see install_torch); MONAI-Physio is then constrained to it.
    No setup runs at import or module discovery. Tests never install packages.
    Dependency conflicts remain errors; protected constraints are not relaxed.
    """
    import slicer

    if slicer.app.testingEnabled():
        return False
    if torch_build and torch_build != installed_torch():
        if not slicer.util.confirmYesNoDisplay(
            f"Replace PyTorch {installed_torch() or '(not installed)'} with "
            f"{torch_build}? This downloads several GB and needs a Slicer "
            "restart afterwards.",
            windowTitle="MONAI Physio setup",
        ):
            return False
        install_torch(torch_build, log)
    else:
        # Keep the installed PyTorch; avoid a torch-scatter source build.
        current = installed_torch()
        try:
            importlib.metadata.version("torch-scatter")
        except importlib.metadata.PackageNotFoundError:
            if current and "+" in current:
                install_torch_scatter(current, log=log)
    constraints = bundled_constraints()
    with tempfile.TemporaryDirectory(prefix="slicer-physio-setup-") as work:
        path = Path(work) / "constraints.txt"
        path.write_text("\n".join(constraints) + "\n", encoding="utf-8")
        if not slicer.util.confirmYesNoDisplay(
            "Install MONAI-Physio and its dependencies into Slicer's Python? "
            "This includes the AI stack and can be a large installation. "
            "Existing numerical package versions will be constrained.",
            windowTitle="MONAI Physio setup",
        ):
            return False
        install_monai_physio(path, slicer.util.pip_install)
    return True
