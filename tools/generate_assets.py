# SPDX-License-Identifier: Apache-2.0
"""Rebuild the original geometric icon and labeled workflow illustration."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def font(size: int) -> ImageFont.ImageFont:
    """Use a scalable font if available; this is an optional contributor tool."""
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def create_assets() -> None:
    """Draw original assets without third-party logos or clinical imagery."""
    icon = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    draw.rounded_rectangle((1, 1, 126, 126), radius=24, fill="#122b3b")
    top, left, right = (64, 20), (24, 43), (104, 43)
    middle, bottom_left, bottom_right, bottom = (64, 66), (24, 88), (104, 88), (64, 111)
    draw.polygon([top, right, middle, left], fill="#8ad6b1")
    draw.polygon([left, middle, bottom, bottom_left], fill="#3aa4b0")
    draw.polygon([middle, right, bottom_right, bottom], fill="#5089cc")
    draw.line([top, left, bottom_left, bottom, bottom_right, right, top],
              fill="#e8f5f5", width=3)
    draw.line([left, middle, right], fill="#e8f5f5", width=3)
    draw.line([middle, bottom], fill="#e8f5f5", width=3)
    icon_path = ROOT / "MONAIPhysioExportUSD/Resources/Icons/MONAIPhysioExportUSD.png"
    icon_path.parent.mkdir(parents=True, exist_ok=True)
    icon.save(icon_path)

    image = Image.new("RGB", (1200, 500), "#f3f7fa")
    draw = ImageDraw.Draw(image)
    draw.text((48, 32), "Slicer model to Omniverse", font=font(34), fill="#142e40")
    draw.text((48, 84), "MONAI Physio extension | Workflow illustration",
              font=font(19), fill="#566c7c")

    def box(bounds: tuple[int, int, int, int], title: str, lines: list[str], color: str) -> None:
        draw.rounded_rectangle(bounds, radius=16, fill="white", outline=color, width=3)
        x, y = bounds[0] + 20, bounds[1] + 18
        draw.text((x, y), title, font=font(23), fill="#142e40")
        for i, line in enumerate(lines):
            draw.text((x, y + 44 + i * 26), line, font=font(17), fill="#486273")

    box((48, 175, 330, 338), "3D Slicer", ["Selected surface model", "Geometry + appearance", "MRML capture on UI thread"], "#3aa4b0")
    box((395, 175, 760, 338), "PythonSlicer worker", ["Same Slicer Python environment", "MONAI-Physio conversion", "UI stays responsive"], "#5089cc")
    box((840, 140, 1152, 258), "Local USD file", [".usd / .usda / .usdc"], "#3aa4b0")
    box((840, 288, 1152, 407), "Omniverse", ["Open or reference the USD"], "#5089cc")
    draw.line([(330, 257), (387, 257)], fill="#486273", width=3)
    draw.polygon([(387, 257), (376, 251), (376, 263)], fill="#486273")
    draw.line([(760, 257), (798, 257), (798, 200), (832, 200)], fill="#486273", width=3)
    draw.polygon([(832, 200), (821, 194), (821, 206)], fill="#486273")
    draw.line([(798, 257), (798, 347), (832, 347)], fill="#486273", width=3)
    draw.polygon([(832, 347), (821, 341), (821, 353)], fill="#486273")
    draw.text((48, 444), "Update or auto-update keeps the USD file current. Open or reference it in Omniverse.",
              font=font(17), fill="#566c7c")
    output = ROOT / "docs/images/workflow.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


if __name__ == "__main__":
    create_assets()
