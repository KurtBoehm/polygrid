# This file is part of https://github.com/KurtBoehm/polygrid.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import io
import subprocess
from pathlib import Path
from typing import TypeVar

import fitz
import cairosvg
import numpy as np
from PIL import Image

from polygrid import PointChainMap, polygonize, svg_paths, tikz_paths

T = TypeVar("T")

# Minimal LaTeX wrapper used to compile the TikZ snippet into a PDF.
_latex_template = """
\\documentclass[tikz]{{standalone}}
\\usepackage{{tikz}}
{header}
\\begin{{document}}
{body}
\\end{{document}}
"""

_base_path = Path(__file__).parent


def wrap_svg(content: str, height: int, width: int) -> str:
    """
    Wrap a fragment of SVG content in a minimal SVG document.

    The resulting SVG has a viewBox of [0, width] × [0, height] and uses
    ``shape-rendering="crispEdges"`` to preserve pixel-aligned rendering.
    """
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        + f'xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {width} {height}" '
        + 'shape-rendering="crispEdges">'
        + content
        + "</svg>"
    )


def svg_to_arr(
    svg: str,
    height: int,
    width: int,
    *,
    monochrome: bool = False,
) -> np.ndarray:
    """
    Render SVG to a NumPy array.

    - RGB: shape ``(height, width, 3)`` with ``dtype=uint8``.
    - Monochrome: shape ``(height, width)`` with ``dtype=bool``.
    """
    png_bytes = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=width,
        output_height=height,
        background_color="white",
    )
    assert isinstance(png_bytes, bytes)
    with Image.open(io.BytesIO(png_bytes)) as img:
        if monochrome:
            return np.array(img.convert("1"), dtype=np.bool_)
        return np.array(img.convert("RGB"), dtype=np.uint8)


def render_tikz(
    chains: PointChainMap[T],
    height: int,
    width: int,
    tmp_path: Path,
    *,
    monochrome: bool = False,
) -> np.ndarray:
    """
    Render TikZ paths (from ``chains``) into a NumPy array.

    - Builds a minimal standalone LaTeX document with a TikZ picture.
    - Runs ``pdflatex``.
    - Renders the first PDF page using PyMuPDF.
    """
    path_map = tikz_paths(chains)

    tex_path = tmp_path / "qr_test.tex"
    colors: list[str] = []
    fills: list[str] = []

    for color, paths in path_map:
        if monochrome:
            # In monochrome mode, the output should only use a single color key.
            assert color == 0
            fills.append(f"  \\fill[even odd rule] {' '.join(paths)};")
        else:
            color_name = f"fill{len(colors)}"
            colors.append(f"\\definecolor{{{color_name}}}{{HTML}}{{{color}}}")
            fills.append(f"  \\fill[fill={color_name}] {' '.join(paths)};")

    tikz_body = "\n".join(fills)
    tikz_env = f"\\begin{{tikzpicture}}\n{tikz_body}\n\\end{{tikzpicture}}"
    tex_source = _latex_template.format(header="\n".join(colors), body=tikz_env)
    tex_path.write_text(tex_source, encoding="utf-8")

    # Run pdflatex in nonstop mode, stop on errors, and write outputs next to the .tex.
    subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            tex_path.parent,
            tex_path,
        ],
        check=True,
    )

    pdf_path = tex_path.with_suffix(".pdf")

    # Render the PDF page so that the width matches the desired pixel width.
    with fitz.Document(pdf_path) as doc:
        page = doc[0]

        # unit: points
        rect: fitz.Rect = page.rect
        pdf_width = rect.width

        # Scale such that resulting image width = desired width.
        scale = width / pdf_width
        pix = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            colorspace=fitz.csGRAY if monochrome else fitz.csRGB,
        )

        buf = np.frombuffer(pix.samples, dtype=np.uint8)
        if monochrome:
            return buf.reshape(pix.height, pix.width) > 128

        return buf.reshape(pix.height, pix.width, 3)


def cat_chains() -> tuple[np.ndarray, PointChainMap[str]]:
    """
    Load the reference image (``cat.png``) and polygonize it.

    :returns: Tuple of ``(arr, chains)``, where ``arr`` is an RGB image and
              ``chains`` are polygon chains grouped by hex color.
    """
    with Image.open(_base_path / "cat.png") as img:
        arr = np.array(img.convert("RGB"), dtype=np.uint8)

    grid = [["{:02x}{:02x}{:02x}".format(*pix) for pix in row] for row in arr]
    chains = polygonize(grid)
    return arr, chains


def test_cat_svg():
    arr, chains = cat_chains()
    height, width, _ = arr.shape

    paths = "".join(
        f'<path fill="#{color}" d="{"".join(paths)}"/>'
        for color, paths in svg_paths(chains, relative=True)
    )
    svg = wrap_svg(paths, height, width)
    svg_arr = svg_to_arr(svg, height, width)

    assert np.array_equal(arr, svg_arr), "SVG render does not match original image."


def test_cat_tikz(tmp_path: Path):
    arr, chains = cat_chains()
    height, width, _ = arr.shape

    tikz_arr = render_tikz(chains, height, width, tmp_path)

    # Small differences (up to 1 per channel) are tolerated.
    diff = np.max(np.abs(arr.astype(int) - tikz_arr.astype(int)))
    assert diff <= 1, f"TikZ render differs from original image by {diff}."


def qr_chains() -> tuple[np.ndarray, PointChainMap[bool]]:
    """Load qr.png and polygonize it."""
    with Image.open(_base_path / "qr.png") as img:
        arr = np.array(img.convert("1"), dtype=np.bool_)

    grid = [[bool(pix) for pix in row] for row in arr]
    chains = polygonize(grid, ignore=lambda pix: pix)
    return arr, chains


def test_qr_svg():
    """SVG polygonization of qr.png matches original pixels."""
    arr, chains = qr_chains()
    height, width = arr.shape

    paths = "".join(
        f'<path fill-rule="evenodd" d="{"".join(paths)}"/>'
        for _, paths in svg_paths(chains, relative=True)
    )
    svg = wrap_svg(paths, height, width)
    svg_arr = svg_to_arr(svg, height, width, monochrome=True)

    assert np.array_equal(arr, svg_arr), "SVG QR render does not match original image."


def test_qr_tikz(tmp_path: Path):
    """TikZ polygonization of qr.png matches original pixels (±1)."""
    arr, chains = qr_chains()
    height, width = arr.shape

    tikz_arr = render_tikz(chains, height, width, tmp_path, monochrome=True)

    assert np.array_equal(arr, tikz_arr), "TikZ QR render differs from original image."
