# 🧩 PolyGrid: Grids as Polygons

PolyGrid converts a 2D grid of values into polygons where each **contiguous region** of equal-valued cells is represented as one or more **merged polygons**, not as a grid of tiny squares.
This **eliminates hideous hairline gaps** between cells within each region and **minimizes the number of points per polygon** for compact output.

PolyGrid can generate:

- **TikZ** paths for LaTeX
- **SVG** paths that are aggressively minimized to save space

The `pytest`-based **test suite with 100% coverage** (for both TikZ and SVG) is available in the [`tests` directory](https://github.com/KurtBoehm/polygrid/blob/main/tests).

[![Tests with 100% coverage](https://github.com/KurtBoehm/polygrid/actions/workflows/test.yml/badge.svg)](https://github.com/KurtBoehm/polygrid/actions/workflows/test.yml)

## 📦 Installation

PolyGrid is available [on PyPI](https://pypi.org/project/polygrid/) and can be installed via `pip`:

```sh
pip install polygrid
```

## 🧩 Core API

The main entry point is `polygonize`, which takes a 2D grid with arbitrary values:

```python
from polygrid import polygonize

grid = [
    [0, 0, 0, 0, 0],
    [0, 0, 1, 2, 0],
    [0, 2, 1, 2, 0],
    [0, 2, 1, 1, 0],
    [0, 0, 0, 0, 0],
]

chains_by_value = polygonize(grid)

for value, chains in chains_by_value.items():
    print(f"{value}: {chains}")
```

Output:

```
0: [[[(0, 0), (5, 0), (5, 5), (0, 5)], [(1, 2), (1, 4), (4, 4), (4, 1), (2, 1), (2, 2)]]]
1: [[[(1, 2), (1, 3), (3, 3), (3, 4), (4, 4), (4, 2)]]]
2: [[[(1, 3), (3, 3), (3, 4), (1, 4)]], [[(2, 1), (4, 1), (4, 2), (2, 2)]]]
```

This example highlights key properties of `polygonize`:

- Cells are grouped into **4-connected regions** using a customizable equality predicate.
- Each distinct cell value maps to a list of **polygon groups**:
  - A polygon group is a list of **closed chains** of integer grid points.
  - If a group has more than one chain, it is intended to be filled using the **even-odd rule**: the first chain is the outer boundary, remaining chains are holes.
- All polygons are **rectilinear**, and **collinear vertices are removed** for compact output.

Connectivity and ignored values are customizable:

```python
chains_by_value = polygonize(
    grid,
    # treat “zero vs non-zero” as the grouping criterion
    equals=lambda a, b: (a == 0) == (b == 0),
    # skip cells with value 0 entirely
    ignore=lambda v: v == 0,
)
```

When defining `equals`, you must ensure that `equals(a, b)` is `True` only when `ignore(a) == ignore(b)`.

The result can be passed directly to the SVG and TikZ helpers described below.

## 🖼️ SVG Output

`svg_paths` turns the polygon chains into very compact SVG `path` data:

```python
from polygrid import polygonize, svg_paths

w, g, b = "white", "green", "black"

grid = [
    [b, b, b, b, b],
    [b, b, g, w, b],
    [b, w, g, w, b],
    [b, w, g, g, b],
    [b, b, b, b, b],
]

chains_by_value = polygonize(grid)

for color, paths in svg_paths(chains_by_value, relative=True):
    for d in paths:
        print(f'<path fill-rule="evenodd" fill="{color}" d="{d}"/>')
```

Output:

```svg
<path fill-rule="evenodd" fill="black" d="M0 0V5H5V0zM2 1H4V4H1V2H2z"/>
<path fill-rule="evenodd" fill="green" d="M2 1H3V3H4V4H2z"/>
<path fill-rule="evenodd" fill="white" d="M3 1V3H4V1z"/>
<path fill-rule="evenodd" fill="white" d="M1 2V4H2V2z"/>
```

Here, each polygon group becomes one SVG `path` with one closed subpath per chain; if there is more than one closed subpath (to represent holes), `fill-rule="evenodd"` must be used.

The generated path data is very compact:

- All segments are axis-aligned and encoded using only `M`, `H`, `V`, and `Z`.
- For each step, absolute vs. relative commands are chosen to minimize output length.
- With `relative=True`, relative moves can be used between successive groups when that shortens the output.

You can transform coordinates via `point_transform`, which must yield numeric coordinates that support subtraction and string formatting; PolyGrid provides minimized formatting for `int`, `float`, and `Decimal`:

```python
paths_by_value = svg_paths(
    chains_by_value,
    # scale coordinates by 1.5
    point_transform=lambda p: (1.5 * p[0], 1.5 * p[1]),
    relative=True,
)
```

The output is suitable for embedding directly into an SVG document:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 21 21">
  <!-- output like that from before -->
</svg>
```

## 🖼️ TikZ Output

`tikz_paths` converts polygon chains into TikZ path specifications:

```python
from polygrid import polygonize, tikz_paths

w, g, b = "white", "green", "black"

grid = [
    [b, b, b, b, b],
    [b, b, g, w, b],
    [b, w, g, w, b],
    [b, w, g, g, b],
    [b, b, b, b, b],
]

chains_by_value = polygonize(grid)

for color, paths in tikz_paths(chains_by_value):
    for path in paths:
        print(f"\\path[even odd rule, fill={color}] {path};")
```

Output:

```latex
\path[even odd rule, fill=black] (0, 0) -- (0, -5) -- (5, -5) -- (5, 0) -- cycle (2, -1) -- (4, -1) -- (4, -4) -- (1, -4) -- (1, -2) -- (2, -2) -- cycle;
\path[even odd rule, fill=green] (2, -1) -- (3, -1) -- (3, -3) -- (4, -3) -- (4, -4) -- (2, -4) -- cycle;
\path[even odd rule, fill=white] (3, -1) -- (3, -3) -- (4, -3) -- (4, -1) -- cycle;
\path[even odd rule, fill=white] (1, -2) -- (1, -4) -- (2, -4) -- (2, -2) -- cycle;
```

Here, each polygon group becomes one TikZ path with one closed subpath per chain.
You can attach any TikZ styles to the generated paths (`rounded corners`, `line join=round`, etc.)—because each connected region is rendered as a single path, such styles apply to the whole region rather than to individual cells.

By default, `tikz_paths` flips the vertical axis so that `y` increases upwards (as in TikZ).
You can override this via `point_transform`:

```python
paths_by_value = tikz_paths(
    chains_by_value,
    # flip vertical axis and scale by 1.5
    point_transform=lambda p: (-1.5 * p[0], 1.5 * p[1]),
)
```

The TikZ output is designed to integrate easily into a `tikzpicture`:

```latex
\begin{tikzpicture}[x=1mm, y=1mm, region/.style={draw=none, even odd rule}]
  % output like that from before, ideally using the “region” style
\end{tikzpicture}
```

## ⚠️ Limitations and Workarounds

PolyGrid is optimized for single-colour regions on a solid background that you ignore (e.g. QR codes, monochrome glyphs, or logos with clean, blocky regions).
In these cases, each region becomes one or a few merged polygons, and there are no internal gaps within a region.

For complex pixel art or images with many adjacent colours, each colour is turned into its own set of polygons that merely share boundaries.
When such polygons are rasterized, normal antialiasing can introduce visible hairline seams between colours, even though the polygons touch exactly.

If hairline gaps are a problem, you can add `shape-rendering="crispEdges"` to the `<svg>` element.
This disables antialiasing of edges and makes the output behave much more like the original grid; the visual effect is essentially that of the source image scaled up with nearest-neighbour interpolation.

## 🧠 Algorithm Overview

PolyGrid converts a 2D grid to merged polygons in two main stages:

1. **Connected components and boundary extraction**:
   - Performs a 4-neighbour BFS flood fill over the grid for each non-ignored value.
   - For every cell in a component, its four unit-square edges are added to a `Counter` in a canonical (sorted-endpoint) form.
   - Edges seen exactly once belong to the region boundary (outer boundary or hole).
2. **Cycle tracing and polygon simplification**:
   - Builds an undirected adjacency graph from the remaining boundary edges.
   - Finds connected components of this boundary graph.
   - For each boundary component, traces a “wall-hugging” cycle:
     - At each step, the walk prefers turning (non-collinear successor) over going straight.
     - This produces visually pleasing outlines with rounded-corner rendering.
   - If the initial cycle does not cover all edges, it is iteratively extended:
     - Additional cycles are constructed that follow any remaining unused edges (again preferring turns) until the component is fully covered.
   - Each cycle is simplified by removing collinear vertices, yielding compact rectilinear polygons that exactly cover the original cells.

The result is a mapping from cell values to polygon groups, ready for SVG or TikZ export.

## 🧪 Testing

PolyGrid includes `pytest`-based tests that cover the entire code base with 100% code coverage.

Development dependencies can be installed via the `dev` extra:

```sh
pip install .[dev]
```

All tests (including coverage reporting via `pytest-cov`) can then be run from the project root:

```sh
pytest --cov
```

The TikZ tests are relatively slow, as they require `pdflatex` to compile a LaTeX document to PDF, which is then rasterized using PyMuPDF.
To reduce test times, the `dev` dependencies include `pytest-xdist`, so tests can be run in parallel:

```sh
pytest --cov -n auto  # or a fixed number of workers
```

## 📜 Licence

This library is licensed under the Mozilla Public Licence 2.0, provided in [`License`](https://github.com/KurtBoehm/polygrid/blob/main/License).
