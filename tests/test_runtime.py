# SPDX-License-Identifier: Apache-2.0
"""Test interpreter selection and package protection without changing Python."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import vtk


path = (
    Path(__file__).resolve().parents[1]
    / "MONAIPhysioExportUSD/MONAIPhysioLib/runtime.py"
)
spec = importlib.util.spec_from_file_location("extension_runtime", path)
assert spec is not None and spec.loader is not None
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RuntimeTests(unittest.TestCase):
    def test_monai_physio_version_check(self) -> None:
        def installed(version):
            def lookup(name):
                if version is None:
                    raise runtime.importlib.metadata.PackageNotFoundError(name)
                return version
            return patch.object(runtime.importlib.metadata, "version", lookup)

        self.assertEqual(runtime.REQUIREMENT, "monai-physio>=2026.11.1")
        for version, problem in (
            (None, "not installed"),
            ("2026.9.1", "2026.9.1 is installed"),
            ("2026.11.0", "2026.11.0 is installed"),
            ("2026.11.1", None),
            ("2027.1.0", None),
        ):
            with self.subTest(version=version), installed(version):
                result = runtime.monai_physio_problem()
                if problem is None:
                    self.assertIsNone(result)
                else:
                    self.assertIn(problem, result)

    def test_finds_python_slicer_without_external_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / "bin").mkdir()
            name = "PythonSlicer.exe" if sys.platform == "win32" else "PythonSlicer"
            executable = home / "bin" / name
            executable.touch()
            app = types.SimpleNamespace(
                slicerHome=str(home), applicationDirPath=lambda: str(home)
            )
            with patch.dict(sys.modules, {"slicer": types.SimpleNamespace(app=app)}):
                self.assertEqual(runtime.python_slicer(), executable)
                executable.unlink()
                with self.assertRaises(FileNotFoundError):
                    runtime.python_slicer()

    def test_constraints_preserve_existing_versions(self) -> None:
        versions = {"numpy": np.__version__, "vtk": vtk.vtkVersion.GetVTKVersion()}

        def version(name: str) -> str:
            if name in versions:
                return versions[name]
            raise runtime.importlib.metadata.PackageNotFoundError(name)

        with patch.object(runtime.importlib.metadata, "version", side_effect=version):
            self.assertEqual(runtime.bundled_constraints(), [
                f"numpy=={np.__version__}", f"vtk=={vtk.vtkVersion.GetVTKVersion()}"
            ])

    def test_missing_distribution_metadata_blocks_setup(self) -> None:
        with patch.object(
            runtime.importlib.metadata, "version",
            side_effect=runtime.importlib.metadata.PackageNotFoundError,
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot protect"):
                runtime.bundled_constraints()

    def test_conflicting_loaded_version_blocks_setup(self) -> None:
        with patch.object(runtime.importlib.metadata, "version", return_value="0.0"):
            with self.assertRaisesRegex(RuntimeError, "differs from pip metadata"):
                runtime.bundled_constraints()

    def test_antspyx_is_installed_without_its_numpy_pin(self) -> None:
        metadata = {
            "monai-physio": [
                "numpy>=1.24", "scipy>=1.10.0", "antspyx>=0.6.1",
                "pytest>=7; extra == 'dev'", "pywin32; sys_platform == 'nowhere'",
            ],
            "antspyx": ["pandas", "numpy<2.4.0", "scipy<1.16", "webcolors>=1"],
        }
        calls: list = []

        def requires(name):
            return metadata[name]

        with patch.object(runtime.importlib.metadata, "requires", requires):
            self.assertEqual(
                runtime.requirements_of("monai-physio", skip=("antspyx",)),
                ["numpy>=1.24", "scipy>=1.10.0"],
            )
            self.assertEqual(
                runtime.requirements_of("antspyx", unpin=("numpy", "scipy")),
                ["pandas", "numpy", "scipy", "webcolors>=1"],
            )
            runtime.install_monai_physio(Path("c.txt"), calls.append)

        constraints = ["-c", "c.txt"]
        self.assertEqual(calls, [
            [*constraints, runtime.REQUIREMENT, "--no-deps"],
            [*constraints, "numpy>=1.24", "scipy>=1.10.0"],
            [*constraints, "antspyx>=0.6.1", "--no-deps"],
            [*constraints, "pandas", "numpy", "scipy", "webcolors>=1"],
        ])
        # No step may hand pip the antspyx pins that conflict with Slicer.
        self.assertFalse(any("<2.4" in a or "<1.16" in a for c in calls for a in c))

    def test_testing_mode_never_installs(self) -> None:
        app = types.SimpleNamespace(testingEnabled=lambda: True)
        with patch.dict(sys.modules, {"slicer": types.SimpleNamespace(app=app)}):
            with patch.object(runtime, "bundled_constraints") as constraints:
                self.assertFalse(runtime.ensure_native_dependencies())
                constraints.assert_not_called()


class TorchChoiceTests(unittest.TestCase):
    """PyTorch menu entries and the install commands they produce."""

    def fake_slicer(self) -> tuple[types.SimpleNamespace, list]:
        calls: list = []
        util = types.SimpleNamespace(
            pip_install=lambda args: calls.append(("pip", list(args))),
            _executePythonModule=lambda m, args: calls.append((m, list(args))),
        )
        return types.SimpleNamespace(util=util), calls

    def test_choices_mark_blackwell_and_missing_scatter_wheels(self) -> None:
        choices = dict((build, label) for label, build in runtime.torch_choices())
        self.assertIn("2.11.0+cu128", choices)
        self.assertIn("Blackwell", choices["2.11.0+cu128"])
        self.assertNotIn("torch-scatter", choices["2.11.0+cu128"])
        # Neither PyG nor this extension's release has a wheel for these.
        self.assertIn("no prebuilt torch-scatter", choices["2.11.0+cu130"])
        self.assertTrue(runtime.has_torch_scatter_wheel("2.12.1+cu126"))
        with self.assertRaises(ValueError):
            runtime._torch_install_arguments("2.12.1+cu128")

    def test_release_wheels_fill_pyg_gaps_on_windows_cp312(self) -> None:
        with patch.object(runtime.sys, "platform", "win32"), patch.object(
            runtime.sys, "version_info", (3, 12, 10)
        ):
            self.assertEqual(
                runtime.release_torch_scatter_wheel("2.14.1+cu132"),
                f"{runtime.RELEASE_URL}/"
                "torch_scatter-2.1.2+pt214cu132-cp312-cp312-win_amd64.whl",
            )
            self.assertIsNone(runtime.release_torch_scatter_wheel("2.11.0+cu130"))
            self.assertTrue(runtime.has_torch_scatter_wheel("2.13.0+cpu"))
            slicer, calls = self.fake_slicer()
            with patch.dict(sys.modules, {"slicer": slicer}):
                self.assertTrue(runtime.install_torch_scatter(
                    "2.14.1+cu132", force=True, log=lambda m: None
                ))
            (_, args), = calls
            self.assertTrue(args[0].endswith("+pt214cu132-cp312-cp312-win_amd64.whl"))
            self.assertIn("--force-reinstall", args)
        with patch.object(runtime.sys, "platform", "linux"):
            self.assertIsNone(runtime.release_torch_scatter_wheel("2.14.1+cu132"))
            self.assertFalse(runtime.has_torch_scatter_wheel("2.14.1+cu132"))

    def test_pip_fallback_forces_then_fills_dependencies(self) -> None:
        slicer, calls = self.fake_slicer()
        modules = {"slicer": slicer, "PyTorchUtils": None}
        with patch.dict(sys.modules, modules):
            sys.modules.pop("torch", None)
            runtime.install_torch("2.11.0+cu128", log=lambda m: None)
        forced, filled, scatter = (args for _, args in calls)
        self.assertEqual(forced[:2], ["torch==2.11.0+cu128", "torchvision==0.26.0+cu128"])
        for flag in ("--force-reinstall", "--no-cache-dir", "--no-deps"):
            self.assertIn(flag, forced)
            self.assertNotIn(flag, filled)
        self.assertIn("https://download.pytorch.org/whl/cu128", forced)
        self.assertIn("https://data.pyg.org/whl/torch-2.11.0+cu128.html", scatter)

    def test_pytorch_extension_installs_with_light_the_torch(self) -> None:
        slicer, calls = self.fake_slicer()
        extension = types.SimpleNamespace(PyTorchUtilsLogic=lambda: types.SimpleNamespace(
            _installLightTheTorch=lambda: None
        ))
        ltt = types.ModuleType("light_the_torch")
        modules = {"slicer": slicer, "PyTorchUtils": extension,
                   "light_the_torch": ltt, "light_the_torch._patch": ltt}
        with patch.dict(sys.modules, modules):
            sys.modules.pop("torch", None)
            runtime.install_torch("2.12.1+cpu", log=lambda m: None)
        (m1, forced), (m2, filled), (pip, scatter) = calls
        self.assertEqual((m1, m2, pip), ("light_the_torch", "light_the_torch", "pip"))
        self.assertEqual(forced[:3], ["install", "torch==2.12.1", "torchvision==0.27.1"])
        self.assertIn("--pytorch-computation-backend=cpu", forced)
        self.assertIn("--force-reinstall", forced)
        self.assertNotIn("--force-reinstall", filled)
        # data.pyg.org names pages by minor version.
        self.assertIn("https://data.pyg.org/whl/torch-2.12.0+cpu.html", scatter)

    def test_loaded_torch_blocks_replacement(self) -> None:
        slicer, calls = self.fake_slicer()
        with patch.dict(sys.modules, {"slicer": slicer, "torch": types.ModuleType("torch")}):
            with self.assertRaisesRegex(RuntimeError, "already loaded"):
                runtime.install_torch("2.11.0+cu128")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
