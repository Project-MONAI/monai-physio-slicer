# SPDX-License-Identifier: Apache-2.0
"""Cardiac simulation worker arguments and label handling, without running the network."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "MONAIPhysioExportUSD/MONAIPhysioLib"
# The worker imports its shared helpers from the respiratory worker beside it.
sys.path.insert(0, str(LIB))
spec = importlib.util.spec_from_file_location(
    "cardiac_simulation_worker", LIB / "cardiac_simulation_worker.py"
)
assert spec is not None and spec.loader is not None
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
# The helpers both workers share, loaded the way the cardiac worker loads them.
import respiratory_simulation_worker as shared  # noqa: E402


class CardiacSimulationWorkerTests(unittest.TestCase):
    def test_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vtp = Path(tmp) / "vtp"
            with patch.object(worker, "run") as run, patch.object(
                worker, "gpu_status"
            ) as gpu, patch.object(worker, "fit_in_child_process") as fit:
                worker.main([
                    "--image", "ct.nrrd", "--weights-dir", tmp, "--vtp-dir", str(vtp),
                ])
            # Simpleware runs outside PyTorch, so the GPU is not probed.
            gpu.assert_not_called()
            run.assert_called_once_with(
                Path("ct.nrrd"), "CT", Path(tmp), vtp, None, None, "slicer",
                worker.SLICER_HEART_COLOR, 10, 60.0, None, "simpleware", None,
                phase="predict",
            )
            self.assertEqual(Path(fit.call_args[0][0]).name, "cardiac_simulation_worker.py")

    def test_all_options_and_stale_result_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vtp = Path(tmp) / "vtp"
            vtp.mkdir()
            (vtp / "result.json").write_text("{}")
            with patch.object(worker, "run") as run, patch.object(
                worker, "gpu_status", return_value="cuda test GPU"
            ), patch.object(worker, "fit_in_child_process"):
                worker.main([
                    "--image", "ct.nrrd", "--name", "Chest", "--weights-dir", tmp,
                    "--vtp-dir", str(vtp), "--ct-dir", "ct", "--usd-dir", "usd",
                    "--usd-colors", "anatomy", "--slicer-color", "0.1", "0.2", "0.3",
                    "--frames", "20", "--cycles-per-minute", "75",
                    "--usd-name", "Beat.usdc", "--segmenter", "nvsegment",
                ])
            self.assertFalse((vtp / "result.json").exists())
            run.assert_called_once_with(
                Path("ct.nrrd"), "Chest", Path(tmp), vtp, Path("ct"), Path("usd"),
                "anatomy", (0.1, 0.2, 0.3), 20, 75.0, "Beat.usdc", "nvsegment", None,
                phase="predict",
            )

    def test_segmenter_requirements(self) -> None:
        import torch

        with self.assertRaisesRegex(FileNotFoundError, "Simpleware Medical was not found"):
            worker.heart_segmenter("simpleware", "C:/missing/Console.exe", 40)
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "NV-Segment-CTMR needs a CUDA GPU"):
                worker.heart_segmenter("nvsegment", None, 40)
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            worker.main(["--image", "ct.nrrd", "--weights-dir", "w", "--vtp-dir", "v",
                         "--segmenter", "totalsegmentator"])

    def test_nvsegment_heart_labels_map_to_simpleware_ids(self) -> None:
        import itk
        import numpy as np

        from monai_physio import SegmentNVSegmentCTMRI

        # Every NV-Segment-CTMR heart label is mapped, and nothing else.
        heart = SegmentNVSegmentCTMRI(log_level=40).taxonomy.labels_in_group("heart")
        self.assertEqual(set(worker.NVSEGMENT_TO_SIMPLEWARE), set(heart))
        source = np.zeros((3, 3, 3), np.uint8)
        source[0, 0, 0] = 151  # ventricle_left -> 1
        source[1, 1, 1] = 154  # ventricle_myocardium_left -> 5
        source[2, 2, 2] = 115  # heart -> 6
        source[0, 2, 2] = 6    # aorta in NV-Segment-CTMR: not heart
        image = itk.image_from_array(source)
        image.SetSpacing([0.5, 0.5, 2.0])
        remapped = worker.remap_nvsegment_heart(image)
        out = itk.array_from_image(remapped)
        self.assertEqual(
            (out[0, 0, 0], out[1, 1, 1], out[2, 2, 2], out[0, 2, 2]), (1, 5, 6, 0)
        )
        self.assertEqual(tuple(remapped.GetSpacing()), (0.5, 0.5, 2.0))

    def test_crop_to_mask_keeps_physical_space(self) -> None:
        import itk
        import numpy as np

        array = np.zeros((40, 50, 60), np.uint8)  # z, y, x
        array[20, 25, 30] = 1
        image = itk.image_from_array(array)
        image.SetSpacing([1.0, 1.0, 2.0])
        image.SetOrigin([10.0, 20.0, 30.0])
        (cropped,) = shared.crop_to_mask([image], array > 0, 4.0)
        # 4 mm is 4 voxels in x and y and 2 in z, around the one masked voxel.
        self.assertEqual(tuple(cropped.GetLargestPossibleRegion().GetSize()), (9, 9, 5))
        center = cropped.TransformPhysicalPointToIndex(
            image.TransformIndexToPhysicalPoint([30, 25, 20])
        )
        self.assertEqual(itk.array_from_image(cropped)[center[2], center[1], center[0]], 1)
        # Clipped at the image border instead of running past it.
        edge = np.zeros_like(array)
        edge[0, 0, 0] = 1
        (cropped,) = shared.crop_to_mask([image], edge > 0, 4.0)
        self.assertEqual(tuple(cropped.GetLargestPossibleRegion().GetSize()), (5, 5, 3))


if __name__ == "__main__":
    unittest.main()
