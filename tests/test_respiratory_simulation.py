# SPDX-License-Identifier: Apache-2.0
"""Respiratory simulation worker arguments, without running the network."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "MONAIPhysioExportUSD/MONAIPhysioLib/respiratory_simulation_worker.py"
spec = importlib.util.spec_from_file_location("respiratory_simulation_worker", path)
assert spec is not None and spec.loader is not None
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class RespiratorySimulationWorkerTests(unittest.TestCase):
    def test_ten_stages(self) -> None:
        self.assertEqual(len(worker.STAGES), 10)
        self.assertEqual(worker.STAGES[0], 0.0)
        self.assertAlmostEqual(worker.STAGES[-1], 0.9)

    def test_optional_outputs_default_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vtp = Path(tmp) / "vtp"
            with patch.object(worker, "run") as run, patch.object(
                worker, "gpu_status", return_value="cuda test GPU"
            ), patch.object(worker, "fit_in_child_process") as fit:
                worker.main([
                    "--image", "ct.nrrd", "--weights-dir", tmp, "--vtp-dir", str(vtp),
                ])
            run.assert_called_once_with(
                Path("ct.nrrd"), "CT", Path(tmp), vtp, None, None, "slicer",
                worker.SLICER_LUNG_COLOR, 10, 12.0, None, phase="predict",
            )
            # The fit runs first, in a child process given the same arguments.
            fit.assert_called_once()
            self.assertEqual(fit.call_args[0][1][:2], ["--image", "ct.nrrd"])

    def test_all_outputs_and_stale_result_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vtp = Path(tmp) / "vtp"
            vtp.mkdir()
            (vtp / "result.json").write_text("{}")
            with patch.object(worker, "run") as run, patch.object(
                worker, "gpu_status", return_value="cuda test GPU"
            ), patch.object(worker, "fit_in_child_process") as fit:
                worker.main([
                    "--image", "ct.nrrd", "--name", "Chest", "--weights-dir", tmp,
                    "--vtp-dir", str(vtp), "--ct-dir", "ct", "--usd-dir", "usd",
                    "--usd-colors", "anatomy", "--slicer-color", "0.1", "0.2", "0.3",
                    "--frames", "20", "--cycles-per-minute", "15",
                    "--usd-name", "Breathing.usdc",
                ])
            self.assertFalse((vtp / "result.json").exists())
            run.assert_called_once_with(
                Path("ct.nrrd"), "Chest", Path(tmp), vtp, Path("ct"), Path("usd"),
                "anatomy", (0.1, 0.2, 0.3), 20, 15.0, "Breathing.usdc", phase="predict",
            )

    def test_segmenter_is_nv_segment_ctmr_and_needs_cuda(self) -> None:
        import torch

        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "NV-Segment-CTMR needs a CUDA GPU"):
                worker.lung_segmenter(20)
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            worker.main(["--image", "ct.nrrd", "--weights-dir", "w", "--vtp-dir", "v",
                         "--segmenter", "totalsegmentator"])

    def test_only_lung_lobes_are_kept(self) -> None:
        from monai_physio import SegmentNVSegmentCTMRI

        lobes = worker.lobe_labels(SegmentNVSegmentCTMRI(log_level=40).taxonomy)
        self.assertEqual(sorted(lobes.values()), [
            "lung_lower_lobe_left", "lung_lower_lobe_right", "lung_middle_lobe_right",
            "lung_upper_lobe_left", "lung_upper_lobe_right",
        ])

    def test_stale_lung_model_warning(self) -> None:
        current = worker.LUNG_MODEL_POINTS
        model = {"components": [[0.0] * (3 * current)] * 2}
        self.assertIsNone(worker.stale_lung_model_warning(Path("m"), current, model))
        # The lung model cached before the current release had 79,978 points.
        old = {"components": [[0.0] * (3 * 79_978)] * 2}
        warning = worker.stale_lung_model_warning(Path("m"), 79_978, old)
        self.assertIsNotNone(warning)
        self.assertIn("Delete that folder", warning or "")
        # A current mean surface with a mismatched PCA model is also stale.
        self.assertIsNotNone(worker.stale_lung_model_warning(Path("m"), current, old))

    def test_cycle_timing(self) -> None:
        self.assertEqual(worker.cycle_stages(4), [0.0, 0.25, 0.5, 0.75])
        # 10 frames per cycle at 12 breaths per minute: one cycle per 5 s.
        self.assertAlmostEqual(worker.frames_per_second(10, 12), 2.0)
        for bad in (1, worker.MAX_FRAMES + 1):
            with self.assertRaises(ValueError):
                worker.cycle_stages(bad)
        with self.assertRaises(ValueError):
            worker.frames_per_second(10, 0)

    def test_allowed_frame_counts_give_distinct_file_tags(self) -> None:
        # MONAI-Physio names stage files s{int(stage * 100):03d}.
        for frames in range(2, worker.MAX_FRAMES + 1):
            tags = [int(s * 100) for s in worker.cycle_stages(frames)]
            self.assertEqual(len(set(tags)), frames, frames)

    def test_unsupported_gpu_falls_back_to_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            worker.os.environ, {}, clear=False
        ):
            worker.os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            with patch.object(worker, "run"), patch.object(
                worker, "gpu_status", return_value="unsupported GPU (sm_120)"
            ), patch.object(worker, "fit_in_child_process"):
                worker.main(["--image", "ct.nrrd", "--weights-dir", tmp,
                             "--vtp-dir", str(Path(tmp) / "vtp")])
            self.assertEqual(worker.os.environ["CUDA_VISIBLE_DEVICES"], "-1")

    def test_phase_progress_maps_fractions(self) -> None:
        import contextlib
        import io

        phases = worker.PhaseProgress()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            phases.phase(60, 90)
            for fraction in (0.0, 0.001, 0.5, 1.0):
                phases(fraction, "Time series")
            phases.phase(90, 100)
            phases(0.0, "USD")
        self.assertEqual(out.getvalue().splitlines(), [
            "PROGRESS 60 Time series",
            "PROGRESS 75 Time series",
            "PROGRESS 90 Time series",
            "PROGRESS 90 USD",
        ])

    def test_fit_phase_runs_in_this_process_and_watches_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vtp = Path(tmp) / "vtp"
            vtp.mkdir()
            # The parent's result is left alone by the child.
            (vtp / "result.json").write_text("{}")
            with patch.object(worker, "run") as run, patch.object(
                worker, "gpu_status"
            ) as gpu, patch.object(worker, "fit_in_child_process") as fit, patch.object(
                worker, "exit_with_parent"
            ) as watch:
                worker.main([
                    "--image", "ct.nrrd", "--weights-dir", tmp, "--vtp-dir", str(vtp),
                    "--phase", "fit",
                ])
            self.assertTrue((vtp / "result.json").exists())
        gpu.assert_not_called()
        fit.assert_not_called()
        watch.assert_called_once()
        self.assertEqual(run.call_args.kwargs, {"phase": "fit"})

    def test_child_process_failure_is_reported(self) -> None:
        with patch("subprocess.call", return_value=3) as call:
            with self.assertRaisesRegex(RuntimeError, "exit code 3"):
                worker.fit_in_child_process(Path("w.py"), ["--image", "ct.nrrd"])
        self.assertEqual(call.call_args[0][0][-4:], ["--image", "ct.nrrd", "--phase", "fit"])

    def test_read_ct_converts_to_signed_short(self) -> None:
        import contextlib
        import io

        import itk
        import numpy as np

        values = np.array([[[-3024, 0, 3071, 40000]]])
        with tempfile.TemporaryDirectory() as tmp:
            for dtype in (np.int32, np.uint16, np.float32, np.int16):
                source = itk.image_from_array(
                    np.clip(values, np.iinfo(np.int16).min if dtype != np.uint16 else 0,
                            np.iinfo(np.int16).max if dtype == np.int16 else None)
                    .astype(dtype)
                )
                source.SetSpacing([0.75, 0.75, 2.5])
                source.SetOrigin([1.0, 2.0, 3.0])
                path = Path(tmp) / f"ct_{np.dtype(dtype).name}.nrrd"
                itk.imwrite(source, str(path))
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    image = worker.read_ct(path)
                array = itk.array_from_image(image)
                self.assertEqual(array.dtype, np.int16, dtype)
                self.assertEqual(tuple(image.GetSpacing()), (0.75, 0.75, 2.5))
                self.assertEqual(tuple(image.GetOrigin()), (1.0, 2.0, 3.0))
                quiet = io.StringIO()
                with contextlib.redirect_stdout(quiet):
                    worker.read_ct(path, quiet=True)
                self.assertEqual(quiet.getvalue(), "")
                if dtype == np.int16:
                    self.assertEqual(out.getvalue(), "")
                else:
                    self.assertIn("Converted the CT voxels", out.getvalue())
                    self.assertEqual(array[0, 0, 2], 3071)
                    # 40000 does not fit in a short and is clipped.
                    self.assertEqual(array[0, 0, 3], 32767)
                    self.assertIn("1 voxels", out.getvalue())

    def test_fit_grid_image_is_cropped_and_isotropic(self) -> None:
        import itk
        import numpy as np

        array = np.zeros((60, 100, 100), np.int16)  # z, y, x at 0.75 x 0.75 x 2.5 mm
        mask = np.zeros(array.shape, bool)
        mask[30, 40:60, 40:60] = True
        image = itk.image_from_array(array)
        image.SetSpacing([0.75, 0.75, 2.5])
        grid = worker.fit_grid_image(image, mask, 10.0, 1.5)
        self.assertEqual(tuple(grid.GetSpacing()), (1.5, 1.5, 1.5))
        # The mask (15 mm in x and y, one slice in z) plus 10 mm on each side.
        size = np.array(grid.GetLargestPossibleRegion().GetSize()) * 1.5
        np.testing.assert_allclose(size, [36.0, 36.0, 25.0], atol=3.0)
        # The original grid would have been 100 x 100 x 60 voxels at 0.75 mm
        # isotropic: 200 x 200 x 200.
        self.assertLess(int(np.prod(grid.GetLargestPossibleRegion().GetSize())), 30 ** 3)

    def test_predict_stages_runs_one_stage_per_call(self) -> None:
        import contextlib
        import io

        calls = []

        class Infer:
            def process_time_series(self, stages, **arguments):
                calls.append((stages, arguments))
                tag = f"s{int(stages[0] * 100):03d}"
                return {
                    "predicted_surfaces": [f"{tag}.vtp"], "warped_images": [f"{tag}.mha"],
                    "stage_meshes": [tag], "transforms": ["large field"],
                }

        with contextlib.redirect_stdout(io.StringIO()):
            result = worker.predict_stages(
                Infer(), [0.0, 0.5], worker.PhaseProgress(), 60, 90, output_directory="o"
            )
        self.assertEqual(calls, [([0.0], {"output_directory": "o"}),
                                 ([0.5], {"output_directory": "o"})])
        self.assertEqual(result, {
            "predicted_surfaces": ["s000.vtp", "s050.vtp"],
            "warped_images": ["s000.mha", "s050.mha"],
            "stage_meshes": ["s000", "s050"],
        })


if __name__ == "__main__":
    unittest.main()
