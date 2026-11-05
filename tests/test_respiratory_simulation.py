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
            ):
                worker.main([
                    "--image", "ct.nrrd", "--weights-dir", tmp, "--vtp-dir", str(vtp),
                ])
            run.assert_called_once_with(
                Path("ct.nrrd"), "CT", Path(tmp), vtp, None, None, "slicer",
                worker.SLICER_LUNG_COLOR, 10, 12.0, None,
            )

    def test_all_outputs_and_stale_result_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vtp = Path(tmp) / "vtp"
            vtp.mkdir()
            (vtp / "result.json").write_text("{}")
            with patch.object(worker, "run") as run, patch.object(
                worker, "gpu_status", return_value="cuda test GPU"
            ):
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
                "anatomy", (0.1, 0.2, 0.3), 20, 15.0, "Breathing.usdc",
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
            ):
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


if __name__ == "__main__":
    unittest.main()
