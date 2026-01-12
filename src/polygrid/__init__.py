# This file is part of https://github.com/KurtBoehm/polygrid.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from collections import Counter, deque
from collections.abc import Iterable
from decimal import Decimal
from typing import Callable, Protocol, Self, TypeVar

__version__ = "1.0.0"

__all__ = ["Point", "Edge", "polygonize", "tikz_paths", "svg_paths"]


class SvgCoordinate(Protocol):
    """Numeric SVG coordinate supporting subtraction."""

    def __sub__(self, value: Self, /) -> Self: ...


T = TypeVar("T")
TikzC = TypeVar("TikzC")
SvgC = TypeVar("SvgC", bound=SvgCoordinate)
SvgInnerC = TypeVar("SvgInnerC", bound=SvgCoordinate)

# 2D integer grid point (row, column).
Point = tuple[int, int]
Edge = tuple[Point, Point]

# A group of polygons that may make use of the even-odd fill rule.
# Each polygon is represented as the chain of points making up its boundary.
PointChains = list[list[list[Point]]]
PointChainMap = dict[T, PointChains]


def normalized_edge(p: Point, q: Point) -> Edge:
    """Canonical representation of an undirected edge with sorted endpoints."""
    return (p, q) if p <= q else (q, p)


def collinear(a: Point, b: Point, c: Point) -> bool:
    """Return whether three grid points are collinear (share a row or a column)."""
    return (a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1])


def connected_components(adj: dict[Point, set[Point]]) -> list[set[Point]]:
    """Compute the connected components of an undirected graph.

    :param adj: Adjacency sets mapping each vertex to its neighbors.
    :returns: One vertex set per connected component.
    """
    unvisited: set[Point] = set(adj)
    components: list[set[Point]] = []

    while unvisited:
        start = unvisited.pop()
        component = {start}
        queue: deque[Point] = deque([start])

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v in unvisited:
                    unvisited.remove(v)
                    component.add(v)
                    queue.append(v)

        components.append(component)

    return components


def polygonize(
    grid: list[list[T]],
    equals: Callable[[T, T], bool] = lambda a, b: a == b,
    ignore: Callable[[T], bool] = lambda _: False,
    remove_collinear: bool = True,
) -> PointChainMap[T]:
    """
    Extract polygon boundaries for all 4-connected regions in a grid.

    The ``grid`` is an ``n × m`` array of cell values. For each distinct value ``v``
    and each 4-connected component of cells with value ``v`` (according to ``equals``)
    that is not ignored by ``ignore``, this function computes its boundary
    on the underlying integer grid.

    A component’s boundary is represented as a list of polygon groups.
    Each group is a list of closed chains, where each chain is a cyclic
    sequence of boundary points. If a group contains multiple chains,
    it uses the even-odd fill rule to represent outer boundaries and holes.

    :param grid:
        Rectangular 2D grid of cell values.
    :param equals:
        Equivalence predicate on cell values used to group cells into regions.
        Default: ``a == b``.
    :param ignore:
        Predicate selecting cell values that should be ignored.
        Ignored cells do not produce polygons. Default: ignore nothing.
    :param remove_collinear:
        If ``True`` (default), drop vertices that are collinear with their neighbors
        to simplify polygons.
        If ``False``, keep all grid corner points.
    :returns:
        Mapping from each distinct, non-ignored value to its polygon groups,
        each of which is a list of closed chains.
    """
    point_chains: PointChainMap[T] = {}

    n, [m] = len(grid), {len(row) for row in grid}

    def neighbors(r: int, c: int) -> Iterable[Point]:
        """Yield 4-connected neighbors of cell ``(r, c)`` within bounds."""
        for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < m:
                yield nr, nc

    visited = [[False] * m for _ in range(n)]

    for r in range(n):
        for c in range(m):
            value = grid[r][c]

            if visited[r][c] or ignore(value):
                continue

            # Flood-fill this connected component and count its cell edges.
            queue: deque[Point] = deque([(r, c)])
            visited[r][c] = True
            edge_counts: Counter[Edge] = Counter()

            while queue:
                cr, cc = queue.popleft()

                # Cell corners (top-left, top-right, bottom-left, bottom-right).
                p00, p01 = (cr, cc), (cr, cc + 1)
                p10, p11 = (cr + 1, cc), (cr + 1, cc + 1)

                # Count every edge of this cell.
                for p, q in ((p00, p01), (p00, p10), (p01, p11), (p10, p11)):
                    edge_counts[normalized_edge(p, q)] += 1

                # Add unvisited neighbors with equal value.
                for nr, nc in neighbors(cr, cc):
                    if not visited[nr][nc] and equals(value, grid[nr][nc]):
                        visited[nr][nc] = True
                        queue.append((nr, nc))

            # Edges used exactly once form the boundary graph (outer and holes).
            boundary_edges = {e for e, cnt in edge_counts.items() if cnt == 1}
            assert boundary_edges

            # Build adjacency list of the boundary graph.
            adj: dict[Point, set[Point]] = {}
            for p, q in boundary_edges:
                adj.setdefault(p, set()).add(q)
                adj.setdefault(q, set()).add(p)

            components = connected_components(adj)
            components.sort(key=len, reverse=True)

            # Find the best cycle for each boundary component (largest to smallest).
            chains: list[list[Point]] = []
            for component in components:
                # Starting vertex is arbitrary → choose lexicographic minimum.
                init = min(component)
                edges_left = {
                    e for e in boundary_edges if e[0] in component or e[1] in component
                }

                # Construct the initial cycle with a preference for edges
                # that are not collinear with the preceding edge (if there is one).
                # This ensures “wall-hugging” behaviour when entering a hole,
                # which leads to more visually pleasing results when rounding
                # corners.
                chain = [init]
                prec: Point | None = None

                while True:
                    curr = chain[-1]
                    closing = normalized_edge(curr, init)
                    if closing in edges_left:
                        edges_left.remove(closing)
                        break

                    # Available edges from curr that are still unused.
                    succs = [
                        v for v in adj[curr] if normalized_edge(curr, v) in edges_left
                    ]

                    if prec is not None:
                        # Prefer a turn (non-collinear) over going straight.
                        pv = prec
                        succs.sort(key=lambda sv: collinear(pv, curr, sv))
                    succ = succs[0]

                    chain.append(succ)
                    edges_left.remove(normalized_edge(curr, succ))
                    prec = curr

                # The cycle does not cover the entire connected component.
                # Extend the cycle by constructing a new cycle that uses
                # edges that are not included in the cycle already, i.e. those
                # in `edges_left`, if available (still preferring turns).
                # If there is no such edge, use the same successor used before.
                while len(set(chain)) < len(component):
                    new_chain = [init]
                    prec = None
                    idx = 1

                    while True:
                        curr = new_chain[-1]
                        succs = [
                            v
                            for v in adj[curr]
                            if normalized_edge(curr, v) in edges_left
                        ]
                        if not succs:
                            if idx == len(chain):
                                break
                            # Follow previous chain when no unused edge is available.
                            succ = chain[idx]
                            idx += 1
                        else:
                            if prec is not None:
                                pv = prec
                                succs.sort(key=lambda sv: collinear(pv, curr, sv))
                            succ = succs[0]
                            edges_left.remove(normalized_edge(curr, succ))

                        new_chain.append(succ)
                        prec = curr

                    chain = new_chain

                if remove_collinear:
                    # Remove collinear vertices to simplify polygons.
                    i = 0
                    while i < len(chain):
                        p0, p1, p2 = chain[i - 1], chain[i], chain[(i + 1) % len(chain)]
                        if collinear(p0, p1, p2):
                            del chain[i]
                        else:
                            i += 1

                chains.append(chain)

            point_chains.setdefault(value, []).append(chains)

    return point_chains


def _key_tikz_paths(
    point_chains: PointChains,
    *,
    point_transform: Callable[[Point], tuple[TikzC, TikzC]] = lambda p: (-p[0], p[1]),
) -> Iterable[str]:
    """
    Yield TikZ path specifications for one value’s polygon groups.

    :param point_chains:
        Polygon groups for a single value as returned by :func:`polygonize`.
    :param point_transform:
        Map from ``(row, column)`` to TikZ coordinates ``(y, x)``.
        The default flips the vertical axis (TikZ ``y`` increases upward) and keeps
        the horizontal coordinate.
    :returns:
        TikZ path strings, one per polygon group.
    """
    for chains in point_chains:
        # Each chain becomes a closed path.
        yield " ".join(
            " -- ".join(f"({c}, {r})" for r, c in (point_transform(p) for p in chain))
            + " -- cycle"
            for chain in chains
        )


def tikz_paths(
    point_chain_map: PointChainMap[T],
    *,
    point_transform: Callable[[Point], tuple[TikzC, TikzC]] = lambda p: (-p[0], p[1]),
) -> Iterable[tuple[T, Iterable[str]]]:
    """
    Convert polygon chains to TikZ path specifications.

    :param point_chain_map:
        Polygon groups per value as returned by :func:`polygonize`.
    :param point_transform:
        Map from ``(row, column)`` to TikZ coordinates ``(y, x)``.
        The default flips the vertical axis (TikZ ``y`` increases upward) and keeps
        the horizontal coordinate.
    :returns:
        ``(value, paths)`` pairs, where ``paths`` yields one TikZ path
        per polygon group.
    """
    for key, point_chains in point_chain_map.items():
        yield key, _key_tikz_paths(point_chains, point_transform=point_transform)


def _format_svg_coord(v: SvgCoordinate):
    """Format a coordinate value for SVG path data."""
    if isinstance(v, Decimal):
        return f"{v.normalize():g}"
    if isinstance(v, (int, float)):
        return f"{v:g}"
    return str(v)


def _key_svg_paths(
    point_chains: PointChains,
    *,
    point_transform: Callable[[Point], tuple[SvgC, SvgC]] = lambda p: p,
    relative: bool,
) -> Iterable[str]:
    """
    Yield SVG path ``d`` attributes for one value’s polygon groups.

    :param point_chains:
        Polygon groups for a single value as returned by :func:`polygonize`.
    :param point_transform:
        Map from grid points to SVG coordinates. Default: identity.
    :param relative:
        Whether to allow relative moves between successive groups.
        If ``True``, each group after the first may start with a relative ``m`` command
        if it is shorter than the corresponding absolute ``M``.
    :returns:
        SVG path ``d`` strings, one per polygon group.
    """
    f = _format_svg_coord

    def move(
        p: tuple[SvgInnerC, SvgInnerC] | None,
        q: tuple[SvgInnerC, SvgInnerC],
    ) -> str:
        qr, qc = q
        abs_cmd = f"M{f(qc)} {f(qr)}"
        if p is None:
            return abs_cmd
        rel_cmd = f"m{f(qc - p[1])} {f(qr - p[0])}"
        return abs_cmd if len(abs_cmd) <= len(rel_cmd) else rel_cmd

    def line(axis: str, src: SvgInnerC, dst: SvgInnerC) -> str:
        abs_arg, rel_arg = f(dst), f(dst - src)
        abs_cmd, rel_cmd = axis.upper() + abs_arg, axis + rel_arg
        return abs_cmd if len(abs_arg) <= len(rel_arg) else rel_cmd

    prev: tuple[SvgC, SvgC] | None = None
    for chains in point_chains:
        parts: list[str] = []
        for chain in chains:
            # Each chain becomes a closed sub-path.
            p0 = point_transform(chain[0])
            parts.append(move(prev, p0))
            r, c = p0

            for nr, nc in (point_transform(p) for p in chain[1:]):
                dr, dc = nr - r, nc - c
                assert dr == 0 or dc == 0, f"{dr} {dc}"
                parts.append(line("h", c, nc) if dr == 0 else line("v", r, nr))
                r, c = nr, nc

            parts.append("z")
            if relative:
                prev = p0

        yield "".join(parts)


def svg_paths(
    point_chain_map: PointChainMap[T],
    *,
    point_transform: Callable[[Point], tuple[SvgC, SvgC]] = lambda p: p,
    relative: bool,
) -> Iterable[tuple[T, Iterable[str]]]:
    """
    Convert polygon chains to SVG path ``d`` attributes.

    :param point_chain_map:
        Polygon groups per value as returned by :func:`polygonize`.
    :param point_transform:
        Map from grid points to SVG coordinates. Default: identity.
    :param relative:
        Whether to allow relative moves between successive groups.
        If ``True``, each group after the first may start with a relative ``m`` command
        if it is shorter than the corresponding absolute ``M``.
    :returns:
        ``(value, paths)`` pairs, where ``paths`` yields one SVG ``d`` string
        per polygon group.
    """
    for key, point_chains in point_chain_map.items():
        polys = _key_svg_paths(
            point_chains,
            point_transform=point_transform,
            relative=relative,
        )
        yield key, polys
