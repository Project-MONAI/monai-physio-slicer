# SPDX-License-Identifier: Apache-2.0
"""Convert a Slicer model snapshot with the public MONAI-Physio converter.

Run with Slicer's PythonSlicer or an external MONAI-Physio Python::

    python export_worker.py model.usdc snapshot.json [snapshot.json ...]
        [--colors slicer|anatomy] [--mesh surface|tetrahedra|hexahedra]
        [--resolution MM]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any, Optional

COLOR_MODES = ("slicer", "anatomy")
MESH_TYPES = ("surface", "tetrahedra", "hexahedra")
DEFAULT_RESOLUTION_MM = 2.0
# Voxels in one model's bounding box at the chosen resolution; finer grids
# run out of memory in the worker.
MAX_VOXELS = 30_000_000

# VTK_HEXAHEDRON corners as (i, j, k) offsets: bottom quad, then the top quad.
HEX_CORNERS = (
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
)
# Hexahedron faces as corner indices, counter-clockwise seen from outside.
HEX_FACES = (
    (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
    (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
)

# monai-physio's import-time install hints (CuPy, uv extras) do not apply to
# Slicer's pip-managed Python; keep them out of the module log.
for _hint in (r"CuPy is not installed", r"monai-physio was installed with pip"):
    warnings.filterwarnings("ignore", message=_hint, category=UserWarning)


def progress(percent: float, message: str) -> None:
    """Report progress to the Slicer module, which drives its progress bar."""
    print(f"PROGRESS {int(round(percent))} {message}", flush=True)


def _display_primvars(mesh: Any, metadata: dict[str, Any]) -> None:
    """Write the captured Slicer colors and opacity as display primvars."""
    import numpy as np
    from pxr import Vt

    display = metadata["display"]
    rgba = metadata.get("rgba")
    interpolation = metadata.get("color_interpolation", "constant")
    if rgba is None:
        rgba = [[*display["color"], 1.0]]
    colors = np.asarray(rgba, dtype=np.float32)
    if colors.ndim != 2 or colors.shape[1] != 4 or not np.isfinite(colors).all():
        raise ValueError("Captured RGBA colors must be a finite N by 4 array.")
    expected = {
        "constant": lambda: 1,
        "vertex": lambda: len(mesh.GetPointsAttr().Get()),
        "uniform": lambda: len(mesh.GetFaceVertexCountsAttr().Get()),
    }
    if interpolation not in expected or len(colors) != expected[interpolation]():
        raise ValueError("Captured colors do not match the exported mesh topology.")
    # VTK display/LUT colors are sRGB; Preview Surface inputs are linear.
    rgb = np.clip(colors[:, :3], 0, 1)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    opacity = np.clip(colors[:, 3] * display["opacity"], 0, 1)
    mesh.CreateDisplayColorPrimvar(interpolation).Set(Vt.Vec3fArray.FromNumpy(linear))
    mesh.CreateDisplayOpacityPrimvar(interpolation).Set(
        Vt.FloatArray.FromNumpy(opacity)
    )


def _slicer_look(stage: Any, display: dict[str, Any], looks: dict[Any, Any]) -> Any:
    """Return a Preview Surface material reading each mesh's display primvars.

    Color and opacity come from per-mesh primvars, so models share one
    material; only distinct Phong specular settings need their own.
    """
    from pxr import Gf, Sdf, UsdShade

    specular = round(float(display["specular"]), 4)
    roughness = round(math.sqrt(2.0 / (float(display["specular_power"]) + 2.0)), 4)
    key = (specular, roughness)
    if key in looks:
        return looks[key]
    name = "SlicerMaterial" if not looks else f"SlicerMaterial_{len(looks)}"
    material = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("Surface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    for primvar, kind, output, input_name in (
        ("displayColor", "float3", "Float3", "diffuseColor"),
        ("displayOpacity", "float", "Float", "opacity"),
    ):
        reader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild(primvar))
        reader.CreateIdAttr(f"UsdPrimvarReader_{kind}")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(primvar)
        result = reader.CreateOutput("result", getattr(Sdf.ValueTypeNames, output))
        input_type = (
            Sdf.ValueTypeNames.Color3f
            if input_name == "diffuseColor"
            else Sdf.ValueTypeNames.Float
        )
        shader.CreateInput(input_name, input_type).ConnectToSource(result)
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(specular)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    surface = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface)
    looks[key] = material
    return material


def _material(
    stage: Any, mesh: Any, metadata: dict[str, Any], looks: dict[Any, Any]
) -> None:
    """Map captured display color/opacity and Phong parameters to USD Preview."""
    from pxr import UsdShade

    _display_primvars(mesh, metadata)
    material = _slicer_look(stage, metadata["display"], looks)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    _display_flags(mesh, metadata["display"])


def _display_flags(mesh: Any, display: dict[str, Any]) -> None:
    from pxr import UsdGeom

    mesh.CreateDoubleSidedAttr(not display["backface_culling"])
    mesh.CreateVisibilityAttr(
        UsdGeom.Tokens.inherited if display["visible"] else UsdGeom.Tokens.invisible
    )


# Slicer (TotalSegmentator terminology) spellings -> MONAI-Physio label words.
_WORD_ALIASES = {
    "vertebrae": "vertebra", "cartilages": "cartilage", "clavicle": "clavicula",
    "venous": "vein", "veins": "vein", "arteries": "artery", "ribs": "rib",
    "trachea": "airway",
}
_FILLER_WORDS = {"and", "of", "the", "system"}


def anatomy_tokens(name: str) -> list[str]:
    """Lower-case words of a model or label name, with spelling aliases."""
    words = [w for w in re.split(r"[^a-z0-9]+", name.lower()) if w]
    return [_WORD_ALIASES.get(w, w) for w in words if w not in _FILLER_WORDS]


def anatomy_candidates(name: str) -> list[str]:
    """Names to try against MONAI-Physio's anatomy render entries.

    Slicer names read "Left Lung" or "left ventricle"; the entries use
    "lung" or "ventricle_left". Try the normalized name, then its words
    reversed, so both orders resolve.
    """
    tokens = anatomy_tokens(name)
    candidates = ["_".join(tokens), "_".join(reversed(tokens))]
    return [c for i, c in enumerate(candidates) if c and c not in candidates[:i]]


def taxonomy_groups() -> dict[frozenset[str], str]:
    """Map each segmenter label's word set to its anatomy group.

    MONAI-Physio's TotalSegmentator taxonomy already groups structures that
    have no render entry of their own (ribs, vertebrae and clavicles in
    "bone"; esophagus and colon in "soft_tissue"). Word sets make "left rib 1"
    match "rib_left_1" and "L2 vertebra" match "vertebrae_L2".
    """
    import logging

    try:
        from monai_physio import SegmentChestTotalSegmentator

        taxonomy = SegmentChestTotalSegmentator(log_level=logging.ERROR).taxonomy
    except Exception:
        return {}
    groups: dict[frozenset[str], str] = {}
    for group in taxonomy.group_names():
        if group == taxonomy.OTHER_GROUP:
            continue
        for label in taxonomy.labels_in_group(group).values():
            words = [w for w in anatomy_tokens(label) if w != "highres"]
            groups.setdefault(frozenset(words), group)
    return groups


def select_anatomy(
    anatomy: Any, name: str, groups: dict[frozenset[str], str]
) -> tuple[str, str]:
    """Pick the anatomy entry for a model name, and say how it was found.

    Organ-level render entries win ("left subclavian artery" -> artery), then
    the segmenter taxonomy group ("left rib 1" -> bone), else "other".
    """
    for candidate in anatomy_candidates(name):
        if anatomy.resolve_anatomy_type(candidate) is not None:
            return candidate, "name"
    group = groups.get(frozenset(anatomy_tokens(name)))
    if group is not None and anatomy.resolve_anatomy_type(group) is not None:
        return group, "taxonomy group"
    return "other", "no match"


def _anatomy_material(
    anatomy: Any, mesh: Any, metadata: dict[str, Any],
    groups: dict[frozenset[str], str],
) -> tuple[str, str]:
    """Bind MONAI-Physio's OmniSurface material for the model's anatomy.

    ProcessUSDAnatomy reuses /World/Looks/OmniSurface_<name> when it exists,
    so models of one anatomy share a material. The display color primvar is
    set to that anatomy's diffuse color so viewers without MDL support match.
    Returns the anatomy entry used and how it was found.
    """
    from pxr import Gf, UsdGeom, Vt

    chosen, how = select_anatomy(anatomy, metadata["name"], groups)
    prim = mesh.GetPrim()
    if prim.IsA(UsdGeom.Mesh):
        anatomy.apply_anatomy_material_to_mesh(str(mesh.GetPath()), chosen)
    else:
        # apply_anatomy_material_to_mesh accepts only UsdGeom.Mesh; a TetMesh
        # takes the same material through the prim-level call.
        anatomy.apply_anatomy_material_to_prim(
            prim, anatomy._resolve_render_params(chosen)
        )
    display = metadata["display"]
    mesh.CreateDisplayColorPrimvar("constant").Set(
        Vt.Vec3fArray([Gf.Vec3f(*anatomy.get_anatomy_diffuse_color(chosen))])
    )
    mesh.CreateDisplayOpacityPrimvar("constant").Set(
        Vt.FloatArray([float(display["opacity"])])
    )
    _display_flags(mesh, display)
    return chosen, how


def _prim_names(names: list[str]) -> list[str]:
    """Unique, valid USD prim names for the model names, in order."""
    from pxr import Tf

    used: set[str] = set()
    result = []
    for name in names:
        token = Tf.MakeValidIdentifier(name) or "Model"
        unique, index = token, 1
        while unique in used:
            index += 1
            unique = f"{token}_{index}"
        used.add(unique)
        result.append(unique)
    return result


def voxelize(polydata: Any, resolution: float) -> tuple[Any, Any]:
    """Return the closed surface's inside as a (z, y, x) mask, and the world
    position of voxel (0, 0, 0)'s center, on an axis-aligned grid of
    resolution-millimeter voxels. A voxel is inside when its center is."""
    import numpy as np
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    if not resolution > 0:
        raise ValueError("The resolution must be a positive number of mm.")
    bounds = np.asarray(polydata.GetBounds(), dtype=np.float64)
    low, high = bounds[0::2], bounds[1::2]
    dims = np.maximum(np.ceil((high - low) / resolution).astype(int), 1)
    if int(np.prod(dims)) > MAX_VOXELS:
        raise ValueError(
            f"A {resolution:g} mm resolution needs {int(np.prod(dims)):,} voxels "
            f"for this model (limit {MAX_VOXELS:,}); use a coarser resolution."
        )
    # Voxel centers, centered on the bounds.
    origin = (low + high) / 2 - (dims - 1) * resolution / 2
    # The stencil reads polygons; strips would be skipped.
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(polydata)
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputConnection(triangles.GetOutputPort())
    stencil.SetOutputOrigin(*origin)
    stencil.SetOutputSpacing(resolution, resolution, resolution)
    stencil.SetOutputWholeExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
    image = vtk.vtkImageStencilToImage()
    image.SetInputConnection(stencil.GetOutputPort())
    image.SetInsideValue(1)
    image.SetOutsideValue(0)
    image.SetOutputScalarTypeToUnsignedChar()
    image.Update()
    scalars = vtk_to_numpy(image.GetOutput().GetPointData().GetScalars())
    return scalars.reshape(dims[::-1]) != 0, origin


def hexahedra(mask: Any, origin: Any, resolution: float) -> tuple[Any, Any]:
    """Return (points, cells) with one VTK_HEXAHEDRON per mask voxel.

    Neighboring voxels share corner points, so the mesh is conforming. Only
    corners of kept voxels become points.
    """
    import numpy as np

    k, j, i = np.nonzero(mask)
    nz, ny, nx = mask.shape
    corner_ids = np.stack(
        [
            (i + di) + (nx + 1) * ((j + dj) + (ny + 1) * (k + dk))
            for di, dj, dk in HEX_CORNERS
        ],
        axis=1,
    )
    used, cells = np.unique(corner_ids, return_inverse=True)
    ci = used % (nx + 1)
    cj = (used // (nx + 1)) % (ny + 1)
    ck = used // ((nx + 1) * (ny + 1))
    corner = np.asarray(origin, dtype=np.float64) - resolution / 2
    points = corner + resolution * np.stack([ci, cj, ck], axis=1)
    return points, cells.reshape(-1, 8)


def _kuhn_tetrahedra() -> Any:
    """Six tetrahedra, as hexahedron corner indices, that fill a cube.

    Each runs from corner (0, 0, 0) to (1, 1, 1) along one ordering of the
    axes, so every cube splits its faces along the same diagonal and the
    tetrahedra of neighboring cubes conform. Each is ordered for a positive
    volume, which USD's rightHanded TetMesh needs for outward faces.
    """
    import itertools

    import numpy as np

    index = {corner: n for n, corner in enumerate(HEX_CORNERS)}
    tetrahedra = []
    for axes in itertools.permutations(range(3)):
        path = [(0, 0, 0)]
        for axis in axes:
            step = list(path[-1])
            step[axis] = 1
            path.append(tuple(step))
        p = np.asarray(path, dtype=np.float64)
        if np.linalg.det(p[1:] - p[0]) < 0:
            path[1], path[2] = path[2], path[1]
        tetrahedra.append([index[c] for c in path])
    return np.asarray(tetrahedra)


def tetrahedra(cells: Any) -> Any:
    """Split each hexahedron of hexahedra() into six conforming tetrahedra."""
    return cells[:, _kuhn_tetrahedra()].reshape(-1, 4)


def boundary_quads(cells: Any) -> Any:
    """Faces of hexahedra() cells that no other cell shares, wound outward."""
    import numpy as np

    faces = cells[:, np.asarray(HEX_FACES)].reshape(-1, 4)
    _, first, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
    )
    return faces[np.sort(first[counts == 1])]


def relax_tetrahedra(
    points: Any, tets: Any, surface: Any, volume_tools: Any
) -> tuple[Any, Any]:
    """Relax voxel tetrahedra onto the model's surface.

    MONAI-Physio's trim_tetrahedra_to_surface drops cells entirely outside,
    then moves the boundary onto the surface while keeping every cell above
    its quality bound, so the mesh follows the smooth surface rather than
    the voxel staircase.
    """
    import numpy as np
    import pyvista as pv

    grid = pv.UnstructuredGrid({pv.CellType.TETRA: tets}, points)
    relaxed = volume_tools.trim_tetrahedra_to_surface(grid, pv.wrap(surface))
    if relaxed.n_cells == 0:
        return np.empty((0, 3)), np.empty((0, 4), dtype=np.int64)
    return (
        np.asarray(relaxed.points, dtype=np.float64),
        np.asarray(relaxed.cells_dict[np.uint8(pv.CellType.TETRA)]),
    )


def _volume_mesh(
    stage: Any, path: str, polydata: Any, mesh_type: str, resolution: float,
    name: str, volume_tools: Any = None,
) -> Any:
    """Replace the surface prim at path with the model's volume mesh.

    Tetrahedra become a UsdGeom.TetMesh, relaxed onto the surface when
    volume_tools (a ProcessVolumetricMesh) is given. USD has no hexahedral prim, so
    hexahedra become a UsdGeom.Mesh of their boundary quads that also holds
    every point, with the cells in the custom int[] attribute
    physio:hexVertexIndices (8 per cell, VTK_HEXAHEDRON order).
    """
    import numpy as np
    from pxr import Sdf, UsdGeom, Vt

    from monai_physio.vtk_to_usd.usd_utils import lps_points_to_usd

    mask, origin = voxelize(polydata, resolution)
    if not mask.any():
        raise ValueError(
            f"'{name}' holds no voxel at {resolution:g} mm, or its surface is "
            "not closed; use a finer resolution."
        )
    points, cells = hexahedra(mask, origin, resolution)
    if mesh_type == "tetrahedra":
        tets = tetrahedra(cells)
        if volume_tools is not None:
            points, tets = relax_tetrahedra(points, tets, polydata, volume_tools)
            if not len(tets):
                raise ValueError(
                    f"Relaxing '{name}' onto its surface removed every cell; "
                    "use a finer resolution."
                )
    stage.RemovePrim(path)
    usd_points = lps_points_to_usd(points)
    extent = UsdGeom.PointBased.ComputeExtent(usd_points)
    if mesh_type == "tetrahedra":
        tets = tets.astype(np.int32)
        mesh = UsdGeom.TetMesh.Define(stage, path)
        mesh.CreatePointsAttr(usd_points)
        mesh.CreateTetVertexIndicesAttr(Vt.Vec4iArray.FromNumpy(tets))
        mesh.CreateSurfaceFaceVertexIndicesAttr(
            UsdGeom.TetMesh.ComputeSurfaceFaces(mesh)
        )
        count = len(tets)
    else:
        quads = boundary_quads(cells).astype(np.int32)
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(usd_points)
        mesh.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(np.full(len(quads), 4, dtype=np.int32))
        )
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(quads.ravel()))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.GetPrim().CreateAttribute(
            "physio:hexVertexIndices", Sdf.ValueTypeNames.IntArray, custom=True
        ).Set(Vt.IntArray.FromNumpy(cells.astype(np.int32).ravel()))
        count = len(cells)
    mesh.CreateExtentAttr(extent)
    prim = mesh.GetPrim()
    prim.SetCustomDataByKey("monai_physio:element_size_mm", float(resolution))
    print(
        f"'{name}': {count:,} {mesh_type} of {resolution:g} mm, "
        f"{len(points):,} points",
        flush=True,
    )
    return mesh


def _convert(
    snapshots: list[Path], destination: Path, colors: str = "slicer",
    mesh_type: str = "surface", resolution: float = DEFAULT_RESOLUTION_MM,
) -> None:
    """Convert version-1 snapshots into one USD stage at destination.

    One ConvertVTKToUSD call writes every model as a sibling mesh under
    /World/SlicerModel, with a single camera and light for the whole scene.
    The converter's per-mesh colormap materials are then replaced by shared
    ones under /World/Looks.

    Input mesh points are world RAS millimeters, a Slicer-specific deviation
    from the library's mesh convention. Parent transforms are already baked.
    Numeric point/cell arrays use ConvertVTKToUSD's standard primvars. Original
    MRML attributes and display parameters are retained as JSON custom data.

    mesh_type "tetrahedra" or "hexahedra" replaces each surface with a volume
    mesh of resolution-millimeter elements (see _volume_mesh); those carry
    one color per model, as Slicer's per-point or per-cell colors do not
    map onto the new cells.
    """
    import pyvista as pv
    import vtk
    from pxr import UsdGeom

    from monai_physio.convert_vtk_to_usd import ConvertVTKToUSD

    metadatas, polydata = [], []
    for snapshot in snapshots:
        metadata = json.loads(snapshot.read_text(encoding="utf-8"))
        if metadata.get("version") != 1:
            raise ValueError("Unsupported Slicer snapshot version.")
        mesh = pv.read(snapshot.parent / metadata["mesh"])
        if not isinstance(mesh, pv.PolyData) or not mesh.n_points or not mesh.n_cells:
            raise ValueError("Snapshot must contain a nonempty polygonal surface.")
        transform = vtk.vtkTransform()
        transform.Scale(-1, -1, 1)
        transformed = vtk.vtkTransformPolyDataFilter()
        transformed.SetInputData(mesh)
        transformed.SetTransform(transform)
        transformed.Update()
        metadatas.append(metadata)
        polydata.append(transformed.GetOutput())
    names = _prim_names([m["name"] for m in metadatas])
    converter = ConvertVTKToUSD(
        data_basename="SlicerModel",
        input_polydata=polydata,
        static_merge=True,
        object_names=names,
        compute_normals=False,
    )
    converter.settings.triangulate_meshes = False
    progress(50, f"Converting {len(names)} model(s) to USD")
    stage = converter.convert(str(destination))
    if mesh_type != "surface":
        volume_tools = None
        if mesh_type == "tetrahedra":
            import logging

            from monai_physio import ProcessVolumetricMesh

            volume_tools = ProcessVolumetricMesh(log_level=logging.WARNING)
        for index, (name, metadata, surface) in enumerate(
            zip(names, metadatas, polydata)
        ):
            progress(
                55 + 15 * index / len(names),
                f"Meshing {mesh_type} ({index + 1} of {len(names)})",
            )
            _volume_mesh(
                stage, f"/World/SlicerModel/{name}", surface, mesh_type,
                resolution, metadata["name"], volume_tools,
            )
    # Materials the converter made from colormaps; replaced below.
    looks_root = stage.GetPrimAtPath("/World/Looks")
    converter_looks = (
        [child.GetPath() for child in looks_root.GetChildren()]
        if looks_root.IsValid() else []
    )

    anatomy = None
    groups: dict[frozenset[str], str] = {}
    if colors == "anatomy":
        from monai_physio import ProcessUSDAnatomy

        anatomy = ProcessUSDAnatomy(stage)
        groups = taxonomy_groups()
    slicer_looks: dict[Any, Any] = {}
    for index, (name, metadata) in enumerate(zip(names, metadatas)):
        progress(
            70 + 25 * index / len(names),
            f"Applying materials ({index + 1} of {len(names)})",
        )
        prim = stage.GetPrimAtPath(f"/World/SlicerModel/{name}")
        if not prim.IsValid() or not prim.IsA(UsdGeom.PointBased):
            raise RuntimeError(f"Expected a model mesh at /World/SlicerModel/{name}.")
        mesh = UsdGeom.PointBased(prim)
        if anatomy is not None:
            chosen, how = _anatomy_material(anatomy, mesh, metadata, groups)
            prim.SetCustomDataByKey("monai_physio:anatomy", chosen)
            material = anatomy.resolve_anatomy_type(chosen)
            print(
                f"'{metadata['name']}' -> material '{material}' "
                f"(entry '{chosen}', by {how})",
                flush=True,
            )
        else:
            if mesh_type != "surface":
                metadata = {k: v for k, v in metadata.items() if k != "rgba"}
            _material(stage, mesh, metadata, slicer_looks)
        prim.SetDisplayName(metadata["name"])
        prim.SetCustomDataByKey(
            "slicer:metadata", json.dumps(metadata, ensure_ascii=True)
        )
    for path in converter_looks:
        stage.RemovePrim(path)
    stage.Save()


def _place(temporary: Path, output: Path, overwrite: bool) -> None:
    if overwrite:
        temporary.replace(output)
    else:
        # Same filesystem: link is atomic and fails if another writer won.
        os.link(temporary, output)


def export_snapshots(
    snapshots: list[Path], output: Path, overwrite: bool = False,
    colors: str = "slicer", mesh_type: str = "surface",
    resolution: float = DEFAULT_RESOLUTION_MM,
) -> Path:
    """Export one or more snapshots to a single USD, atomically replacing only
    when authorized. Models become sibling meshes under /World/SlicerModel.

    colors is "slicer" (each model's Slicer display color via USD Preview
    Surface) or "anatomy" (MONAI-Physio's anatomy material for its name).
    mesh_type is "surface", or "tetrahedra" or "hexahedra" of resolution mm."""
    if colors not in COLOR_MODES:
        raise ValueError(f"colors must be one of {COLOR_MODES}.")
    if mesh_type not in MESH_TYPES:
        raise ValueError(f"mesh_type must be one of {MESH_TYPES}.")
    if not resolution > 0:
        raise ValueError("The resolution must be a positive number of mm.")
    if not snapshots:
        raise ValueError("No model snapshots to export.")
    if output.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("Output must end in .usd, .usda, or .usdc.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="slicer-usd-", dir=output.parent) as work:
        temporary = Path(work) / output.name
        _convert(snapshots, temporary, colors, mesh_type, resolution)
        _place(temporary, output, overwrite)
    return output


def export_snapshot(snapshot: Path, output: Path, overwrite: bool = False) -> Path:
    """Export a single version-1 snapshot; see export_snapshots."""
    return export_snapshots([snapshot], output, overwrite)


def main(argv: Optional[list[str]] = None) -> None:
    """External-worker CLI for the Slicer module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("snapshots", type=Path, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--colors", choices=COLOR_MODES, default="slicer")
    parser.add_argument("--mesh", choices=MESH_TYPES, default="surface")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION_MM,
                        help="Element size in mm for tetrahedra or hexahedra")
    args = parser.parse_args(argv)
    progress(5, "Loading MONAI-Physio")
    output = export_snapshots(
        args.snapshots, args.output, args.overwrite, args.colors,
        args.mesh, args.resolution,
    )
    print(f"USD saved: {output}", flush=True)
    progress(100, "Done")


if __name__ == "__main__":
    import sys

    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    # Work is done; skip interpreter teardown, where background threads from
    # heavy dependencies can keep the process (and the Slicer GUI) waiting.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
