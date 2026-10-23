"""Genotype utilities for ThriftNAS.

``Genotype`` is a namedtuple with four fields::

    normal         – list of 8 (op_name, input_index) tuples (2 per intermediate node)
    normal_concat  – indices of intermediate node outputs that are concatenated [2,3,4,5]
    reduce         – same for reduction cells
    reduce_concat  – [2,3,4,5]

Public API
----------
derive_genotype, save_genotype, load_genotype, plot_genotype,
plot_side_by_side, count_ops
"""

from __future__ import annotations

import json
import os
from collections import Counter, namedtuple
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from thriftnas.ops.primitives import PRIMITIVES

# ---------------------------------------------------------------------------
# Genotype namedtuple
# ---------------------------------------------------------------------------

Genotype = namedtuple("Genotype", ["normal", "normal_concat", "reduce", "reduce_concat"])

# 4 intermediate nodes; their output indices relative to cell states are 2,3,4,5
_CONCAT    = list(range(2, 6))   # [2, 3, 4, 5]
_NUM_NODES = 4
_ZERO_IDX  = PRIMITIVES.index("zero")


# ---------------------------------------------------------------------------
# Core derive algorithm  (spec §7 reference solution, verbatim logic)
# ---------------------------------------------------------------------------

def _derive(alpha: torch.Tensor) -> list[tuple[str, int]]:
    """Derive a discrete gene list from a (14, 8) alpha tensor.

    For each of the 4 intermediate nodes:
    1. Softmax the alpha rows that correspond to this node's incoming edges.
    2. For each edge *j*, compute the best **non-zero** op softmax weight.
    3. Keep the 2 edges with the highest best weight (ranked; ties broken by
       index after ``sorted``).
    4. Record ``(PRIMITIVES[k], j)`` for the best non-zero op on each kept edge.

    Returns a flat list of 8 ``(op_name, input_index)`` tuples.
    """
    w: torch.Tensor = F.softmax(alpha.float(), dim=-1).detach().cpu()
    nz  = _ZERO_IDX
    gene: list[tuple[str, int]] = []
    start, n = 0, 2  # node 0 has 2 incoming edges

    for _ in range(_NUM_NODES):
        W = w[start: start + n]   # shape (n, len(PRIMITIVES))

        # Best non-zero op softmax weight per incoming edge
        best = [
            max(float(W[j, k]) for k in range(len(PRIMITIVES)) if k != nz)
            for j in range(n)
        ]

        # Keep top-2 edges, sorted by edge index (ascending) for reproducibility
        keep = sorted(sorted(range(n), key=lambda j: -best[j])[:2])

        for j in keep:
            k = max(
                (k for k in range(len(PRIMITIVES)) if k != nz),
                key=lambda k: float(W[j, k]),
            )
            gene.append((PRIMITIVES[k], j))

        start += n
        n     += 1

    return gene


def derive_genotype(
    alpha_normal: torch.Tensor,
    alpha_reduce: torch.Tensor,
) -> Genotype:
    """Derive a :class:`Genotype` from raw alpha logits.

    Parameters
    ----------
    alpha_normal:
        Shape ``(14, 8)`` – architecture logits for normal cells.
    alpha_reduce:
        Shape ``(14, 8)`` – architecture logits for reduction cells.

    Returns
    -------
    Genotype
        Named-tuple with ``normal``, ``normal_concat``, ``reduce``,
        ``reduce_concat`` fields.
    """
    return Genotype(
        normal        = _derive(alpha_normal),
        normal_concat = _CONCAT[:],
        reduce        = _derive(alpha_reduce),
        reduce_concat = _CONCAT[:],
    )


# ---------------------------------------------------------------------------
# JSON save / load  (atomic write)
# ---------------------------------------------------------------------------

def save_genotype(genotype: Genotype, path: str | Path) -> None:
    """Atomically save *genotype* to a JSON file at *path*."""
    data: dict[str, Any] = {
        "normal":        [list(t) for t in genotype.normal],
        "normal_concat": genotype.normal_concat,
        "reduce":        [list(t) for t in genotype.reduce],
        "reduce_concat": genotype.reduce_concat,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, str(path))


def load_genotype(path: str | Path) -> Genotype:
    """Load a :class:`Genotype` from the JSON file written by :func:`save_genotype`."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return Genotype(
        normal        = [tuple(t) for t in data["normal"]],
        normal_concat = data["normal_concat"],
        reduce        = [tuple(t) for t in data["reduce"]],
        reduce_concat = data["reduce_concat"],
    )


# ---------------------------------------------------------------------------
# Op counting
# ---------------------------------------------------------------------------

def count_ops(genotype: Genotype) -> dict[str, int]:
    """Count occurrences of each operation across both cell types.

    Returns
    -------
    dict
        Mapping ``op_name -> count``.  Values sum to 16
        (8 ops per cell type × 2 cell types).
    """
    all_ops = [op for op, _ in genotype.normal] + [op for op, _ in genotype.reduce]
    return dict(Counter(all_ops))


# ---------------------------------------------------------------------------
# Graphviz diagram
# ---------------------------------------------------------------------------

def _build_gv_cell(gene: list[tuple[str, int]], title: str):  # -> graphviz.Digraph
    """Build a graphviz Digraph for one cell.  Caller handles rendering."""
    import graphviz as gv

    g = gv.Digraph(
        name=title,
        graph_attr={"label": title, "rankdir": "TB",
                    "ranksep": "0.8", "fontsize": "12", "fontname": "Helvetica"},
        node_attr={"style": "filled", "fontsize": "10", "fontname": "Helvetica"},
        edge_attr={"fontsize": "9",   "fontname": "Helvetica"},
    )

    # Input nodes (same rank)
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("in0", label="c_{k-2}", shape="oval",  fillcolor="#d5e8d4")
        s.node("in1", label="c_{k-1}", shape="oval",  fillcolor="#d5e8d4")

    # Intermediate nodes
    for i in range(_NUM_NODES):
        g.node(f"nd{i}", label=f"node {i}", shape="rect", fillcolor="#dae8fc")

    # Concat output
    g.node("out", label="concat", shape="oval", fillcolor="#fff2cc")

    # Gene edges
    def _src(j: int) -> str:
        if j == 0: return "in0"
        if j == 1: return "in1"
        return f"nd{j - 2}"

    for i in range(_NUM_NODES):
        for op, src_j in gene[2 * i: 2 * i + 2]:
            g.edge(_src(src_j), f"nd{i}", label=op, color="steelblue")

    # Concat dashed edges
    for i in range(_NUM_NODES):
        g.edge(f"nd{i}", "out", style="dashed", arrowhead="none", color="gray")

    return g


def _render_graphviz(gene: list[tuple[str, int]], title: str, out_path: Path) -> bool:
    """Try graphviz render.  Returns True on success, False otherwise."""
    try:
        g = _build_gv_cell(gene, title)
        stem = str(out_path.with_suffix(""))
        g.render(stem, format="png", cleanup=True)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Matplotlib fallback diagram
# ---------------------------------------------------------------------------

def _draw_cell_matplotlib(
    gene: list[tuple[str, int]],
    title: str,
    out_path: Path,
) -> None:
    """Draw a clean cell diagram using matplotlib (no system graphviz needed)."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    FIG_W, FIG_H = 9, 7
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#f8f8f8")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # Positions (x, y in axes coords)
    INPUT_Y  = 0.92
    NODE_Y   = [0.72, 0.52, 0.34, 0.16]
    CONCAT_Y = 0.02
    NODE_X   = [0.22, 0.40, 0.62, 0.80]
    INPUT_X  = [0.32, 0.68]
    CONCAT_X = 0.50

    positions: dict[str, tuple[float, float]] = {
        "in0": (INPUT_X[0], INPUT_Y),
        "in1": (INPUT_X[1], INPUT_Y),
        "out": (CONCAT_X,   CONCAT_Y),
    }
    for i in range(_NUM_NODES):
        positions[f"nd{i}"] = (NODE_X[i], NODE_Y[i])

    def _draw_node(key: str, label: str, color: str) -> None:
        x, y = positions[key]
        ax.text(x, y, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", fc=color, ec="#555", lw=1.2))

    _draw_node("in0", "c_{k-2}", "#d5e8d4")
    _draw_node("in1", "c_{k-1}", "#d5e8d4")
    _draw_node("out", "concat",  "#fff2cc")
    for i in range(_NUM_NODES):
        _draw_node(f"nd{i}", f"node {i}", "#dae8fc")

    def _src_key(j: int) -> str:
        return "in0" if j == 0 else ("in1" if j == 1 else f"nd{j-2}")

    kw_arr = dict(transform=ax.transAxes, xycoords="axes fraction",
                  textcoords="axes fraction")

    # Gene edges
    for i in range(_NUM_NODES):
        for op, src_j in gene[2 * i: 2 * i + 2]:
            sx, sy = positions[_src_key(src_j)]
            dx, dy = positions[f"nd{i}"]
            ax.annotate("", xy=(dx, dy), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.4),
                        **kw_arr)
            ax.text((sx + dx) / 2 + 0.01, (sy + dy) / 2, op,
                    transform=ax.transAxes, ha="left", va="center",
                    fontsize=7, color="#003d73",
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))

    # Concat edges (dashed)
    for i in range(_NUM_NODES):
        sx, sy = positions[f"nd{i}"]
        dx, dy = positions["out"]
        ax.annotate("", xy=(dx, dy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-", color="#999", lw=1.0,
                                    linestyle="dashed"),
                    **kw_arr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_genotype(genotype: Genotype, out_dir: str | Path) -> None:
    """Save ``cell_normal.png`` and ``cell_reduce.png`` in *out_dir*.

    Renders with graphviz if the system binary is available, otherwise falls
    back to a clean matplotlib diagram.

    Parameters
    ----------
    genotype:
        The discrete architecture to visualise.
    out_dir:
        Directory in which to save the two PNG files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = [
        (genotype.normal, "Normal Cell",    "cell_normal"),
        (genotype.reduce, "Reduction Cell", "cell_reduce"),
    ]
    for gene, title, stem in cells:
        out_path = out_dir / f"{stem}.png"
        ok = _render_graphviz(gene, title, out_path)
        if not ok:
            _draw_cell_matplotlib(gene, title, out_path)


def plot_side_by_side(
    genotype_a: Genotype,
    genotype_b: Genotype,
    labels: Sequence[str],
    path: str | Path,
) -> None:
    """Save a 2×2 side-by-side comparison PNG to *path*.

    Layout::

        | labels[0] Normal   | labels[1] Normal   |
        | labels[0] Reduction | labels[1] Reduction |

    Parameters
    ----------
    genotype_a, genotype_b:
        Genotypes to compare.
    labels:
        Two strings used as headings (e.g. ``["lam=0", "lam=1"]``).
    path:
        Destination PNG path.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor("#f4f4f4")
    fig.suptitle(f"{labels[0]}  vs  {labels[1]}", fontsize=14, fontweight="bold")

    combos = [
        (genotype_a.normal, f"{labels[0]} – Normal",    axes[0, 0]),
        (genotype_b.normal, f"{labels[1]} – Normal",    axes[0, 1]),
        (genotype_a.reduce, f"{labels[0]} – Reduction", axes[1, 0]),
        (genotype_b.reduce, f"{labels[1]} – Reduction", axes[1, 1]),
    ]
    for gene, title, ax in combos:
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
        lines = [f"  ({op:<22}  in={j})" for (op, j) in gene]
        ax.text(0.05, 0.96, "\n".join(lines),
                transform=ax.transAxes, va="top", fontsize=8.5,
                fontfamily="monospace",
                bbox=dict(fc="white", ec="#ccc", pad=6, boxstyle="round"))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
