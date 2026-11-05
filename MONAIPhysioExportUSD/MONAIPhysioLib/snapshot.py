# SPDX-License-Identifier: Apache-2.0
"""Slicer-only snapshot capture; no MONAI, PyVista or USD dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import vtk


def capture_model(model: Any, directory: Path) -> Path:
    """Capture one surface model without mutating its mesh or MRML nodes.

    World RAS coordinates are serialized for the external MONAI-Physio worker.
    Unsupported display modes fail explicitly instead of silently changing
    appearance. Phong lighting parameters are retained for approximate mapping.
    """
    import slicer

    if model is None or not model.IsA("vtkMRMLModelNode"):
        raise ValueError("Select a Slicer model node.")
    source = model.GetPolyData()
    display = model.GetDisplayNode()
    if source is None or not source.GetNumberOfPoints() or display is None:
        raise ValueError("The model needs a polygonal surface and display node.")
    if source.GetNumberOfLines() or source.GetNumberOfVerts():
        raise ValueError(
            "Only surface models are supported, not lines or point clouds."
        )
    if display.GetRepresentation() != 2:
        raise ValueError("Switch the model's display representation to Surface.")
    if display.GetTextureImageDataConnection() is not None:
        raise ValueError("Texture images are not supported by this exporter yet.")
    if display.GetClipping():
        raise ValueError("Disable model clipping before export.")
    if display.GetFrontfaceCulling():
        raise ValueError("Front-face culling is not supported; disable it for export.")
    if model.GetNumberOfDisplayNodes() != 1 or display.GetNumberOfViewNodeIDs():
        raise ValueError("Use one display node visible in all 3D views for export.")

    transform = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
        model.GetParentTransformNode(), None, transform
    )
    world = vtk.vtkTransformPolyDataFilter()
    world.SetInputData(source)
    world.SetTransform(transform)
    world.Update()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(world.GetOutputPort())
    triangles.Update()
    poly = vtk.vtkPolyData()
    poly.DeepCopy(triangles.GetOutput())
    if not poly.GetNumberOfPolys():
        raise ValueError("The model contains no surface faces.")
    names = vtk.vtkStringArray()
    model.GetAttributeNames(names)
    attributes = {names.GetValue(i): model.GetAttribute(names.GetValue(i))
                  for i in range(names.GetNumberOfValues())}
    metadata: dict[str, Any] = {
        "version": 1,
        "mesh": "model.vtp",
        "name": model.GetName() or "Model",
        "node_id": model.GetID(),
        "attributes": attributes,
        "parent_transform_id": model.GetTransformNodeID(),
        "display": {
            "color": list(display.GetColor()),
            "opacity": display.GetOpacity(),
            "visible": bool(display.GetVisibility() and display.GetVisibility3D()),
            "backface_culling": bool(display.GetBackfaceCulling()),
            "ambient": display.GetAmbient(),
            "diffuse": display.GetDiffuse(),
            "specular": display.GetSpecular(),
            "specular_power": display.GetPower(),
            "lighting": bool(display.GetLighting()),
            "interpolation": display.GetInterpolation(),
            "scalar_visibility": bool(display.GetScalarVisibility()),
            "scalar_range": list(display.GetScalarRange()),
            "scalar_range_flag": display.GetScalarRangeFlag(),
            "active_scalar": display.GetActiveScalarName(),
        },
        "warnings": [
            "Lighting, ambient/diffuse coefficients and Phong specularity are "
            "renderer-dependent; Preview Surface is an approximation."
        ],
    }
    # Explicit normals preserve Slicer's flat/smooth interpolation choice.
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.SetSplitting(display.GetInterpolation() == 0)
    normals.SetFeatureAngle(0 if display.GetInterpolation() == 0 else 180)
    normals.ComputePointNormalsOn()
    normals.ConsistencyOff()
    normals.Update()
    # Splitting flat faces changes vertex count; capture scalar colors afterward.
    poly.DeepCopy(normals.GetOutput())
    if display.GetScalarVisibility():
        _capture_colors(poly, display, metadata)
    directory.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(directory / "model.vtp"))
    writer.SetInputData(poly)
    if writer.Write() != 1:
        raise OSError("Could not write model snapshot.")
    path = directory / "snapshot.json"
    path.write_text(json.dumps(metadata, ensure_ascii=True), encoding="utf-8")
    return path


def _capture_colors(poly: Any, display: Any, metadata: dict[str, Any]) -> None:
    """Bake Slicer's selected point/cell lookup-table colors into the snapshot."""
    name = display.GetActiveScalarName()
    location = display.GetActiveAttributeLocation()
    if location not in (vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                        vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS):
        raise ValueError("Only point or cell scalar coloring is supported.")
    point_colors = location == vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
    arrays = poly.GetPointData() if point_colors else poly.GetCellData()
    values = arrays.GetArray(name) if name else arrays.GetScalars()
    if values is None:
        raise ValueError("The selected display scalar array is missing.")
    flag = display.GetScalarRangeFlag()
    color_node = display.GetColorNode()
    lut = color_node.GetScalarsToColors() if color_node else None
    if lut is None and flag != display.UseDirectMapping:
        raise ValueError("The scalar display requires a color lookup table.")
    lookup = lut.NewInstance() if lut else vtk.vtkLookupTable()
    if lut:
        lookup.DeepCopy(lut)
    if flag == display.UseDataScalarRange:
        lookup.SetRange(values.GetRange())
    elif flag == display.UseManualScalarRange:
        lookup.SetRange(display.GetScalarRange())
    elif flag == display.UseDataTypeScalarRange:
        lookup.SetRange(values.GetDataTypeMin(), values.GetDataTypeMax())
    mode = (
        vtk.VTK_COLOR_MODE_DIRECT_SCALARS
        if flag == display.UseDirectMapping
        else vtk.VTK_COLOR_MODE_MAP_SCALARS
    )
    colors = lookup.MapScalars(values, mode, -1)
    metadata["rgba"] = [[v / 255.0 for v in colors.GetTuple4(i)]
                        for i in range(colors.GetNumberOfTuples())]
    metadata["color_interpolation"] = "vertex" if point_colors else "uniform"
