# SPDX-License-Identifier: Apache-2.0
"""Small synthetic surfaces test coordinate arithmetic and color association.

VTK executes the real geometry pipeline; only MRML access and remote transport
are mocked. USD tests require the full MONAI-Physio environment.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import vtk


ROOT = Path(__file__).resolve().parents[1]


def load_file(name: str, path: Path) -> Any:
    """Load standalone helpers without importing the AI dependency stack."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


snapshot_module = load_file(
    "ModelSnapshot",
    ROOT / "MONAIPhysioExportUSD/MONAIPhysioLib/snapshot.py",
)
worker = load_file(
    "slicer_export_worker", ROOT / "MONAIPhysioExportUSD/MONAIPhysioLib/export_worker.py"
)


def model_fixture() -> tuple[Any, Any, Any]:
    """A single quad with a world translation; no images or external data."""
    plane = vtk.vtkPlaneSource()
    plane.SetOrigin(10, 20, 30)
    plane.SetPoint1(20, 20, 30)
    plane.SetPoint2(10, 40, 30)
    plane.Update()
    poly = vtk.vtkPolyData()
    poly.DeepCopy(plane.GetOutput())
    display = Mock()
    defaults = {
        "Representation": 2, "TextureImageDataConnection": None, "Clipping": 0,
        "FrontfaceCulling": 0, "NumberOfViewNodeIDs": 0,
        "Color": (0.8, 0.3, 0.1), "Opacity": 0.4, "Visibility": 1,
        "Visibility3D": 1, "BackfaceCulling": 1, "Ambient": 0.0,
        "Diffuse": 1.0, "Specular": 0.2, "Power": 20,
        "Lighting": 1, "Interpolation": 1, "ScalarVisibility": 0,
        "ScalarRange": (0.0, 1.0), "ScalarRangeFlag": 0,
        "ActiveScalarName": "value", "ActiveAttributeLocation": 0,
        "ColorNode": None,
    }
    for key, value in defaults.items():
        getattr(display, "Get" + key).return_value = value
    for index, name in enumerate(("UseManualScalarRange", "UseDataScalarRange",
                                  "UseColorNodeScalarRange", "UseDataTypeScalarRange",
                                  "UseDirectMapping")):
        setattr(display, name, index)
    model = Mock()
    model.IsA.return_value = True
    model.GetPolyData.return_value = poly
    model.GetDisplayNode.return_value = display
    model.GetNumberOfDisplayNodes.return_value = 1
    model.GetAttributeNames.side_effect = lambda a: a.InsertNextValue("anatomy")
    model.GetAttribute.return_value = "heart"
    model.GetName.return_value = "Left ventricle / test"
    model.GetID.return_value = "vtkMRMLModelNode1"
    model.GetTransformNodeID.return_value = "Transform1"
    transform = vtk.vtkTransform()
    transform.Translate(100, 200, 300)
    model.GetParentTransformNode.return_value = transform
    return model, display, poly


class CaptureTests(unittest.TestCase):
    """Verify real VTK geometry capture independently of Slicer's GUI."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        api = types.SimpleNamespace(
            GetTransformBetweenNodes=lambda parent, target, result: (
                result.Concatenate(parent)
            )
        )
        self.fake_slicer = types.SimpleNamespace(vtkMRMLTransformNode=api)
        self.patch = patch.dict(sys.modules, {"slicer": self.fake_slicer})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.model, self.display, self.poly = model_fixture()

    def capture(self) -> tuple[dict[str, Any], Any]:
        """Capture, reload the VTP and return the metadata and geometry."""
        path = snapshot_module.capture_model(self.model, self.path)
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(self.path / "model.vtp"))
        reader.Update()
        return json.loads(path.read_text()), reader.GetOutput()

    def test_world_transform_and_source_unchanged(self) -> None:
        before = self.poly.GetPoint(0)
        metadata, result = self.capture()
        self.assertEqual(result.GetPoint(0), (110.0, 220.0, 330.0))
        self.assertEqual(self.poly.GetPoint(0), before)
        self.assertEqual(result.GetNumberOfPolys(), 2)
        self.assertEqual(metadata["attributes"], {"anatomy": "heart"})
        self.assertEqual(metadata["display"]["opacity"], 0.4)

    def test_non_linear_transform(self) -> None:
        source = vtk.vtkPoints()
        target = vtk.vtkPoints()
        for p, q in [((0, 0, 0), (0, 0, 0)), ((100, 0, 0), (100, 0, 0)),
                     ((0, 100, 0), (0, 100, 0)), ((0, 0, 100), (20, 0, 100))]:
            source.InsertNextPoint(p)
            target.InsertNextPoint(q)
        warp = vtk.vtkThinPlateSplineTransform()
        warp.SetSourceLandmarks(source)
        warp.SetTargetLandmarks(target)
        warp.SetBasisToR()
        self.model.GetParentTransformNode.return_value = warp
        _, result = self.capture()
        expected = warp.TransformPoint(self.poly.GetPoint(0))
        for actual, wanted in zip(result.GetPoint(0), expected):
            self.assertAlmostEqual(actual, wanted, places=4)

    def test_cell_colors_follow_triangulation(self) -> None:
        values = vtk.vtkFloatArray()
        values.SetName("value")
        values.InsertNextValue(0.5)
        self.poly.GetCellData().AddArray(values)
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(2)
        lut.Build()
        lut.SetTableValue(0, 1, 0, 0, 0.5)
        lut.SetTableValue(1, 0, 1, 0, 0.8)
        self.display.GetColorNode.return_value = types.SimpleNamespace(
            GetScalarsToColors=lambda: lut
        )
        self.display.GetActiveAttributeLocation.return_value = 1
        self.display.GetScalarVisibility.return_value = 1
        metadata, result = self.capture()
        self.assertEqual(metadata["color_interpolation"], "uniform")
        self.assertEqual(len(metadata["rgba"]), result.GetNumberOfPolys())
        self.assertEqual(metadata["rgba"][0], metadata["rgba"][1])
        self.assertAlmostEqual(metadata["rgba"][0][3], 0.8, places=2)

    def test_direct_rgba_without_lookup_table(self) -> None:
        values = vtk.vtkUnsignedCharArray()
        values.SetName("value")
        values.SetNumberOfComponents(4)
        for _ in range(self.poly.GetNumberOfPoints()):
            values.InsertNextTuple4(255, 0, 127, 128)
        self.poly.GetPointData().AddArray(values)
        self.display.GetScalarVisibility.return_value = 1
        self.display.GetScalarRangeFlag.return_value = self.display.UseDirectMapping
        metadata, result = self.capture()
        self.assertEqual(len(metadata["rgba"]), result.GetNumberOfPoints())
        self.assertEqual(metadata["rgba"][0], [1, 0, 127 / 255, 128 / 255])

    def test_unsupported_display_fails_explicitly(self) -> None:
        for getter, value, message in (
            ("GetRepresentation", 1, "Surface"),
            ("GetTextureImageDataConnection", object(), "Texture"),
            ("GetClipping", 1, "clipping"),
            ("GetFrontfaceCulling", 1, "Front-face"),
        ):
            with self.subTest(getter=getter):
                method = getattr(self.display, getter)
                previous = method.return_value
                method.return_value = value
                with self.assertRaisesRegex(ValueError, message):
                    self.capture()
                method.return_value = previous



@unittest.skipUnless(
    importlib.util.find_spec("pyvista") and importlib.util.find_spec("pxr"),
    "Requires PyVista, OpenUSD and the installed MONAI-Physio checkout",
)
class UsdRoundTripTests(unittest.TestCase):
    """Verify the real public converter, persisted USD values and shader binding."""

    setUp = CaptureTests.setUp
    capture = CaptureTests.capture

    @classmethod
    def setUpClass(cls) -> None:
        # setUp patches sys.modules, which would discard (and half-reload) the
        # heavy imports made inside a test; import them once, unpatched.
        import monai_physio.convert_vtk_to_usd  # noqa: F401

    def test_usd_round_trip(self) -> None:
        from pxr import Usd, UsdGeom, UsdShade

        self.capture()
        output = worker.export_snapshot(
            self.path / "snapshot.json", self.path / "out.usdc"
        )
        stage = Usd.Stage.Open(str(output))
        meshes = [UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        self.assertEqual(len(meshes), 1)
        mesh = meshes[0]
        for actual, expected in zip(
            mesh.GetPointsAttr().Get()[0], (-0.110, 0.330, 0.220)
        ):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertAlmostEqual(mesh.GetDisplayOpacityPrimvar().Get()[0], 0.4, places=6)
        self.assertEqual(UsdGeom.GetStageMetersPerUnit(stage), 1.0)
        self.assertEqual(UsdGeom.GetStageUpAxis(stage), "Y")
        self.assertFalse(mesh.GetDoubleSidedAttr().Get())
        self.assertTrue(UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0])
        self.assertEqual(mesh.GetPrim().GetDisplayName(), "Left ventricle / test")
        with self.assertRaises(FileExistsError):
            worker.export_snapshot(self.path / "snapshot.json", output)

    def test_anatomy_materials(self) -> None:
        from pxr import Usd, UsdGeom, UsdShade

        self.capture()
        output = worker.export_snapshots(
            [self.path / "snapshot.json"], self.path / "anatomy.usdc",
            colors="anatomy",
        )
        stage = Usd.Stage.Open(str(output))
        mesh = next(UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
        # Fixture name "Left ventricle / test" resolves via "ventricle_left".
        self.assertEqual(
            mesh.GetPrim().GetCustomDataByKey("monai_physio:anatomy"),
            "test_ventricle_left",
        )
        material = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
        self.assertIn("OmniSurface", material.GetPath().name)
        self.assertEqual(len(mesh.GetDisplayColorPrimvar().Get()), 1)

    def test_multiple_models_in_one_file(self) -> None:
        from pxr import Usd, UsdGeom, UsdShade

        self.capture()
        second = self.path / "second"
        second.mkdir()
        for name in ("snapshot.json", "model.vtp"):
            (second / name).write_bytes((self.path / name).read_bytes())
        output = worker.export_snapshots(
            [self.path / "snapshot.json", second / "snapshot.json"],
            self.path / "multi.usdc",
        )
        stage = Usd.Stage.Open(str(output))
        meshes = [UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        self.assertEqual(len(meshes), 2)
        self.assertEqual(len({str(m.GetPath()) for m in meshes}), 2)
        for mesh in meshes:
            self.assertTrue(UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0])

    def capture_cube(self) -> None:
        """Capture a closed 10 mm cube in place of the open fixture quad."""
        cube = vtk.vtkCubeSource()
        cube.SetBounds(0, 10, 0, 10, 0, 10)
        cube.Update()
        self.model.GetPolyData.return_value = cube.GetOutput()
        self.capture()

    def test_tetrahedra(self) -> None:
        from pxr import Usd, UsdGeom, UsdShade

        self.capture_cube()
        output = worker.export_snapshots(
            [self.path / "snapshot.json"], self.path / "tets.usda",
            mesh_type="tetrahedra", resolution=2.0,
        )
        stage = Usd.Stage.Open(str(output))
        meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.TetMesh)]
        self.assertEqual(len(meshes), 1)
        mesh = UsdGeom.TetMesh(meshes[0])
        import numpy as np

        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        tets = np.asarray(mesh.GetTetVertexIndicesAttr().Get())
        # Relaxing onto the surface drops cells lying wholly on the cube's
        # faces, so up to six per voxel remain, all positively oriented.
        self.assertTrue(0 < len(tets) <= 6 * 125)
        p = points[tets]
        self.assertTrue((np.linalg.det(p[:, 1:] - p[:, :1]) > 0).all())
        self.assertGreater(len(mesh.GetSurfaceFaceVertexIndicesAttr().Get()), 0)
        self.assertTrue(UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0])
        self.assertEqual(len(mesh.GetDisplayColorPrimvar().Get()), 1)
        self.assertEqual(
            meshes[0].GetCustomDataByKey("monai_physio:element_size_mm"), 2.0
        )

    def test_tetrahedra_anatomy_materials(self) -> None:
        from pxr import Usd, UsdGeom, UsdShade

        self.capture_cube()
        output = worker.export_snapshots(
            [self.path / "snapshot.json"], self.path / "tets_anatomy.usdc",
            colors="anatomy", mesh_type="tetrahedra", resolution=2.0,
        )
        stage = Usd.Stage.Open(str(output))
        mesh = next(p for p in stage.Traverse() if p.IsA(UsdGeom.TetMesh))
        material = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
        self.assertIn("OmniSurface", material.GetPath().name)

    def test_hexahedra(self) -> None:
        from pxr import Usd, UsdGeom

        self.capture_cube()
        output = worker.export_snapshots(
            [self.path / "snapshot.json"], self.path / "hexes.usdc",
            mesh_type="hexahedra", resolution=2.0,
        )
        stage = Usd.Stage.Open(str(output))
        mesh = next(UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
        self.assertEqual(len(mesh.GetPointsAttr().Get()), 6**3)
        self.assertEqual(set(mesh.GetFaceVertexCountsAttr().Get()), {4})
        self.assertEqual(len(mesh.GetFaceVertexCountsAttr().Get()), 150)
        cells = mesh.GetPrim().GetAttribute("physio:hexVertexIndices").Get()
        self.assertEqual(len(cells), 8 * 125)

    def test_open_surface_has_no_volume(self) -> None:
        self.capture()  # the fixture's single quad encloses nothing
        with self.assertRaisesRegex(ValueError, "no voxel"):
            worker.export_snapshots(
                [self.path / "snapshot.json"], self.path / "open.usdc",
                mesh_type="hexahedra", resolution=2.0,
            )


@unittest.skipUnless(importlib.util.find_spec("numpy"), "Requires NumPy")
class VolumeMeshTests(unittest.TestCase):
    """Voxel hexahedra and their tetrahedra: counts, orientation, conformity."""

    def mesh(self, source: Any, resolution: float) -> tuple[Any, Any]:
        source.Update()
        mask, origin = worker.voxelize(source.GetOutput(), resolution)
        return worker.hexahedra(mask, origin, resolution)

    @staticmethod
    def volumes(points: Any, tets: Any) -> Any:
        import numpy as np

        p = points[tets]
        return np.linalg.det(p[:, 1:] - p[:, :1]) / 6

    def test_cube(self) -> None:
        import numpy as np

        cube = vtk.vtkCubeSource()
        cube.SetBounds(-5, 5, 0, 10, 20, 30)
        points, hexes = self.mesh(cube, 2.0)
        self.assertEqual(hexes.shape, (125, 8))
        self.assertEqual(len(points), 216)
        np.testing.assert_allclose(points.min(axis=0), (-5, 0, 20))
        np.testing.assert_allclose(points.max(axis=0), (5, 10, 30))
        tets = worker.tetrahedra(hexes)
        volumes = self.volumes(points, tets)
        self.assertTrue((volumes > 0).all())
        self.assertAlmostEqual(volumes.sum(), 1000.0)
        # Conforming: every triangle is shared by two tetrahedra, except the
        # boundary's two per voxel face.
        faces = np.sort(tets[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]]
                        .reshape(-1, 3), axis=1)
        _, counts = np.unique(faces, axis=0, return_counts=True)
        self.assertEqual(set(counts), {1, 2})
        self.assertEqual((counts == 1).sum(), 300)

    def test_boundary_quads_face_outward(self) -> None:
        import numpy as np

        cube = vtk.vtkCubeSource()
        cube.SetBounds(0, 6, 0, 6, 0, 6)
        points, hexes = self.mesh(cube, 2.0)
        quads = worker.boundary_quads(hexes)
        self.assertEqual(len(quads), 54)
        q = points[quads]
        normals = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
        outward = q.mean(axis=1) - points.mean(axis=0)
        self.assertTrue(((normals * outward).sum(axis=1) > 0).all())

    def test_sphere_volume(self) -> None:
        import math

        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(20)
        sphere.SetThetaResolution(64)
        sphere.SetPhiResolution(64)
        points, hexes = self.mesh(sphere, 1.0)
        self.assertAlmostEqual(
            len(hexes) / (4 / 3 * math.pi * 20**3), 1.0, delta=0.03
        )

    @unittest.skipUnless(
        importlib.util.find_spec("monai_physio"), "Requires MONAI-Physio"
    )
    def test_relaxation_follows_surface(self) -> None:
        import logging

        import numpy as np
        import pyvista as pv
        from monai_physio import ProcessVolumetricMesh

        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(20)
        sphere.SetThetaResolution(96)
        sphere.SetPhiResolution(96)
        points, hexes = self.mesh(sphere, 2.0)
        surface = sphere.GetOutput()
        points, tets = worker.relax_tetrahedra(
            points, worker.tetrahedra(hexes), surface,
            ProcessVolumetricMesh(log_level=logging.WARNING),
        )
        self.assertTrue((self.volumes(points, tets) > 0).all())
        self.assertAlmostEqual(
            self.volumes(points, tets).sum() / pv.wrap(surface).volume, 1.0,
            delta=0.02,
        )
        boundary = pv.UnstructuredGrid({pv.CellType.TETRA: tets}, points)
        boundary = boundary.extract_surface(algorithm="dataset_surface")
        distance = vtk.vtkImplicitPolyDataDistance()
        distance.SetInput(surface)
        errors = [abs(distance.EvaluateFunction(p)) for p in boundary.points]
        # The voxel staircase alone is about 0.75 mm off on average.
        self.assertLess(float(np.mean(errors)), 0.2)

    def test_resolution_limit(self) -> None:
        cube = vtk.vtkCubeSource()
        cube.SetBounds(0, 1000, 0, 1000, 0, 1000)
        cube.Update()
        with self.assertRaisesRegex(ValueError, "coarser"):
            worker.voxelize(cube.GetOutput(), 0.1)


class AnatomyNameTests(unittest.TestCase):
    """Slicer model names map onto MONAI-Physio taxonomy spellings."""

    def test_candidates(self) -> None:
        self.assertEqual(worker.anatomy_candidates("Left Lung"), ["left_lung", "lung_left"])
        self.assertEqual(
            worker.anatomy_candidates("left ventricle"),
            ["left_ventricle", "ventricle_left"],
        )
        self.assertEqual(worker.anatomy_candidates("heart"), ["heart"])
        self.assertEqual(worker.anatomy_candidates("  "), [])
        self.assertEqual(
            worker.anatomy_candidates("pulmonary venous system"),
            ["pulmonary_vein", "vein_pulmonary"],
        )

    def test_taxonomy_group_fallback(self) -> None:
        class Anatomy:
            entries = {"bone": "Bone", "vein": "Vein", "other": "Other"}

            def resolve_anatomy_type(self, name: str) -> Any:
                return next((v for k, v in self.entries.items() if k in name), None)

        groups = {
            frozenset(worker.anatomy_tokens(label)): group
            for label, group in (
                ("rib_left_1", "bone"), ("vertebrae_L2", "bone"),
                ("clavicula_left", "bone"), ("esophagus", "soft_tissue"),
            )
        }
        anatomy = Anatomy()
        self.assertEqual(
            worker.select_anatomy(anatomy, "left rib 1", groups),
            ("bone", "taxonomy group"),
        )
        self.assertEqual(
            worker.select_anatomy(anatomy, "L2 vertebra", groups)[0], "bone"
        )
        self.assertEqual(
            worker.select_anatomy(anatomy, "left clavicle", groups)[0], "bone"
        )
        # Organ-level entries win over the taxonomy group.
        self.assertEqual(
            worker.select_anatomy(anatomy, "portal vein and splenic vein", groups),
            ("portal_vein_splenic_vein", "name"),
        )
        # A group without a render entry falls back to "other".
        self.assertEqual(
            worker.select_anatomy(anatomy, "esophagus", groups),
            ("other", "no match"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
