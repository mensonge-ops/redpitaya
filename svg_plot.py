"""Lightweight SVG plotting utilities for environments without matplotlib."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass
class Series:
    x: Sequence[float]
    y: Sequence[float]
    color: str = "#1f77b4"
    label: Optional[str] = None
    width: float = 2.0


@dataclass
class Scatter:
    x: Sequence[float]
    y: Sequence[float]
    color: str = "#ffffff"
    edge: str = "#000000"
    size: float = 4.0
    label: Optional[str] = None


@dataclass
class FillRegion:
    x0: float
    x1: float
    color: str = "#66bb6a"
    opacity: float = 0.2


@dataclass
class Panel:
    series: List[Series]
    xlabel: str
    ylabel: str
    title: str = ""
    logx: bool = False
    logy: bool = False
    scatter: List[Scatter] = field(default_factory=list)
    fills: List[FillRegion] = field(default_factory=list)


def _safe_log10(value: float) -> float:
    return math.log10(max(value, 1e-30))


def _format_linear(value: float) -> str:
    if value == 0:
        return "0"
    abs_val = abs(value)
    if abs_val >= 1000 or abs_val < 1e-2:
        return f"{value:.1e}"
    if abs_val < 1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def _generate_linear_ticks(vmin: float, vmax: float, count: int = 5) -> List[Tuple[float, str]]:
    if math.isclose(vmin, vmax):
        return [(vmin, _format_linear(vmin))]
    step = (vmax - vmin) / max(count - 1, 1)
    ticks = [vmin + step * i for i in range(count)]
    return [(tick, _format_linear(tick)) for tick in ticks]


def _generate_log_ticks(vmin: float, vmax: float) -> List[Tuple[float, str]]:
    exp_min = math.floor(vmin)
    exp_max = math.ceil(vmax)
    return [(exp, f"1e{exp}") for exp in range(exp_min, exp_max + 1)]


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#" + "".join(f"{max(0, min(255, c)) :02x}" for c in rgb)


def _interpolate_color(ratio: float, start: str = "#440154", end: str = "#fde725") -> str:
    ratio = max(0.0, min(1.0, ratio))
    s = _hex_to_rgb(start)
    e = _hex_to_rgb(end)
    interp = tuple(int(s[i] + (e[i] - s[i]) * ratio) for i in range(3))
    return _rgb_to_hex(interp)


def save_panel_plots(panels: List[Panel], path: Path, title: Optional[str] = None, width: int = 800, height: int = 600) -> None:
    margin_left = 70
    margin_right = 20
    margin_top = 50
    margin_bottom = 60
    panel_height = (height - margin_top - margin_bottom) / max(len(panels), 1)
    plot_width = width - margin_left - margin_right

    elements: List[str] = []
    if title:
        elements.append(
            f'<text x="{width / 2:.1f}" y="{margin_top / 2:.1f}" font-family="sans-serif" font-size="18" '
            f'text-anchor="middle">{title}</text>'
        )

    legend_entries: List[Tuple[str, str]] = []

    for idx, panel in enumerate(panels):
        y_start = margin_top + idx * panel_height
        plot_height = panel_height - 40
        y_offset = y_start + 20

        transformed_series = []
        all_x: List[float] = []
        all_y: List[float] = []
        for series in panel.series:
            if not series.x or not series.y:
                continue
            if panel.logx:
                x_vals = [_safe_log10(v) for v in series.x]
            else:
                x_vals = list(series.x)
            if panel.logy:
                y_vals = [_safe_log10(max(v, 1e-30)) for v in series.y]
            else:
                y_vals = list(series.y)
            transformed_series.append((x_vals, y_vals, series))
            all_x.extend(x_vals)
            all_y.extend(y_vals)
            if series.label:
                legend_entries.append((series.color, series.label))

        for scatter in panel.scatter:
            if panel.logx:
                all_x.extend(_safe_log10(v) for v in scatter.x)
            else:
                all_x.extend(scatter.x)
            if panel.logy:
                all_y.extend(_safe_log10(max(v, 1e-30)) for v in scatter.y)
            else:
                all_y.extend(scatter.y)

        if not all_x or not all_y:
            continue

        x_min = min(all_x)
        x_max = max(all_x)
        y_min = min(all_y)
        y_max = max(all_y)
        if math.isclose(x_min, x_max):
            x_max = x_min + 1.0
        if math.isclose(y_min, y_max):
            y_max = y_min + 1.0

        def to_pixel(x_val: float, y_val: float) -> Tuple[float, float]:
            px = margin_left + (x_val - x_min) / (x_max - x_min) * plot_width
            py = y_offset + plot_height - (y_val - y_min) / (y_max - y_min) * plot_height
            return px, py

        # background
        elements.append(
            f'<rect x="{margin_left}" y="{y_offset}" width="{plot_width}" height="{plot_height}" '
            f'fill="none" stroke="#444" stroke-width="1" />'
        )

        # fills
        for fill in panel.fills:
            if panel.logx:
                x0 = _safe_log10(fill.x0)
                x1 = _safe_log10(fill.x1)
            else:
                x0, x1 = fill.x0, fill.x1
            px0, _ = to_pixel(x0, y_min)
            px1, _ = to_pixel(x1, y_min)
            width_fill = abs(px1 - px0)
            x_fill = min(px0, px1)
            elements.append(
                f'<rect x="{x_fill}" y="{y_offset}" width="{width_fill}" height="{plot_height}" '
                f'fill="{fill.color}" fill-opacity="{fill.opacity}" stroke="none" />'
            )

        # axes ticks
        if panel.logx:
            ticks_x = _generate_log_ticks(x_min, x_max)
        else:
            ticks_x = _generate_linear_ticks(x_min, x_max)
        if panel.logy:
            ticks_y = _generate_log_ticks(y_min, y_max)
        else:
            ticks_y = _generate_linear_ticks(y_min, y_max)

        for tick_val, label in ticks_x:
            px, _ = to_pixel(tick_val, y_min)
            elements.append(f'<line x1="{px}" y1="{y_offset + plot_height}" x2="{px}" y2="{y_offset + plot_height + 6}" stroke="#333" />')
            elements.append(
                f'<text x="{px}" y="{y_offset + plot_height + 22}" font-family="sans-serif" font-size="12" text-anchor="middle">{label}</text>'
            )

        for tick_val, label in ticks_y:
            _, py = to_pixel(x_min, tick_val)
            elements.append(f'<line x1="{margin_left - 6}" y1="{py}" x2="{margin_left}" y2="{py}" stroke="#333" />')
            elements.append(
                f'<text x="{margin_left - 10}" y="{py + 4}" font-family="sans-serif" font-size="12" text-anchor="end">{label}</text>'
            )

        # axis labels
        elements.append(
            f'<text x="{margin_left + plot_width / 2:.1f}" y="{y_offset + plot_height + 40}" font-family="sans-serif" font-size="14" text-anchor="middle">{panel.xlabel}</text>'
        )
        elements.append(
            f'<text x="{margin_left - 50}" y="{y_offset + plot_height / 2:.1f}" font-family="sans-serif" font-size="14" '
            f'text-anchor="middle" transform="rotate(-90 {margin_left - 50},{y_offset + plot_height / 2:.1f})">{panel.ylabel}</text>'
        )

        if panel.title:
            elements.append(
                f'<text x="{margin_left + plot_width / 2:.1f}" y="{y_offset - 6}" font-family="sans-serif" font-size="15" text-anchor="middle">{panel.title}</text>'
            )

        # lines
        for x_vals, y_vals, series in transformed_series:
            points = [to_pixel(x, y) for x, y in zip(x_vals, y_vals)]
            path_d = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
            elements.append(
                f'<polyline points="{path_d}" fill="none" stroke="{series.color}" stroke-width="{series.width}" stroke-linejoin="round" stroke-linecap="round" />'
            )

        # scatter points
        for scatter in panel.scatter:
            if panel.logx:
                sx = [_safe_log10(v) for v in scatter.x]
            else:
                sx = list(scatter.x)
            if panel.logy:
                sy = [_safe_log10(max(v, 1e-30)) for v in scatter.y]
            else:
                sy = list(scatter.y)
            for x_val, y_val in zip(sx, sy):
                px, py = to_pixel(x_val, y_val)
                r = scatter.size
                elements.append(
                    f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{r}" fill="{scatter.color}" stroke="{scatter.edge}" stroke-width="1" />'
                )
            if scatter.label:
                legend_entries.append((scatter.edge, scatter.label))

    # legend
    if legend_entries:
        unique_entries = []
        for color, label in legend_entries:
            if all(existing[1] != label for existing in unique_entries):
                unique_entries.append((color, label))
        legend_x = width - margin_right - 140
        legend_y = margin_top
        for idx, (color, label) in enumerate(unique_entries):
            y = legend_y + idx * 20
            elements.append(f'<rect x="{legend_x}" y="{y - 10}" width="14" height="14" fill="{color}" stroke="#000" stroke-width="0.5" />')
            elements.append(
                f'<text x="{legend_x + 20}" y="{y + 2}" font-family="sans-serif" font-size="12">{label}</text>'
            )

    svg_content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        *elements,
        "</svg>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg_content), encoding="utf-8")


def save_heatmap(
    x_values: Sequence[float],
    y_values: Sequence[float],
    matrix: Sequence[Sequence[float]],
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str = "Intensity",
    scatter: Optional[List[Tuple[float, float]]] = None,
) -> None:
    width = 820
    height = 520
    margin_left = 80
    margin_right = 80
    margin_top = 60
    margin_bottom = 70

    ncols = len(x_values)
    nrows = len(y_values)
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    if ncols == 0 or nrows == 0:
        return

    cell_width = plot_width / ncols
    cell_height = plot_height / nrows

    flat_values = [value for row in matrix for value in row]
    data_min = min(flat_values)
    data_max = max(flat_values)
    span = data_max - data_min if data_max != data_min else 1.0

    elements: List[str] = [
        f'<text x="{width / 2:.1f}" y="{margin_top / 2:.1f}" font-family="sans-serif" font-size="18" text-anchor="middle">{title}</text>'
    ]

    for row_idx, y_val in enumerate(y_values):
        for col_idx, x_val in enumerate(x_values):
            value = matrix[row_idx][col_idx]
            ratio = (value - data_min) / span
            color = _interpolate_color(ratio)
            x0 = margin_left + col_idx * cell_width
            y0 = margin_top + (nrows - row_idx - 1) * cell_height
            elements.append(
                f'<rect x="{x0}" y="{y0}" width="{cell_width}" height="{cell_height}" fill="{color}" stroke="none" />'
            )

    # axes rectangle
    elements.append(
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#333" stroke-width="1" />'
    )

    def _format_ticks(values: Sequence[float], count: int = 6) -> List[Tuple[int, str]]:
        if not values:
            return []
        step = max(1, len(values) // max(count - 1, 1))
        ticks = list(range(0, len(values), step))
        if ticks[-1] != len(values) - 1:
            ticks.append(len(values) - 1)
        return [(idx, _format_linear(values[idx])) for idx in ticks]

    for idx, label in _format_ticks(x_values):
        px = margin_left + idx * cell_width + cell_width / 2
        elements.append(f'<line x1="{px}" y1="{margin_top + plot_height}" x2="{px}" y2="{margin_top + plot_height + 6}" stroke="#333" />')
        elements.append(
            f'<text x="{px}" y="{margin_top + plot_height + 24}" font-family="sans-serif" font-size="12" text-anchor="middle">{label}</text>'
        )

    for idx, label in _format_ticks(y_values):
        py = margin_top + (nrows - idx - 1) * cell_height + cell_height / 2
        elements.append(f'<line x1="{margin_left - 6}" y1="{py}" x2="{margin_left}" y2="{py}" stroke="#333" />')
        elements.append(
            f'<text x="{margin_left - 10}" y="{py + 4}" font-family="sans-serif" font-size="12" text-anchor="end">{label}</text>'
        )

    # axis labels
    elements.append(
        f'<text x="{margin_left + plot_width / 2:.1f}" y="{height - margin_bottom / 2:.1f}" font-family="sans-serif" font-size="14" text-anchor="middle">{xlabel}</text>'
    )
    elements.append(
        f'<text x="{margin_left - 50}" y="{margin_top + plot_height / 2:.1f}" font-family="sans-serif" font-size="14" text-anchor="middle" transform="rotate(-90 {margin_left - 50},{margin_top + plot_height / 2:.1f})">{ylabel}</text>'
    )

    # scatter overlay
    if scatter:
        for x_val, y_val in scatter:
            if x_val not in x_values or y_val not in y_values:
                continue
            col = x_values.index(x_val)
            row = y_values.index(y_val)
            px = margin_left + col * cell_width + cell_width / 2
            py = margin_top + (nrows - row - 1) * cell_height + cell_height / 2
            elements.append(
                f'<circle cx="{px}" cy="{py}" r="6" fill="#ffffff" stroke="#000000" stroke-width="1" />'
            )

    # colorbar
    bar_x = width - margin_right + 20
    bar_y = margin_top
    bar_height = plot_height
    bar_width = 14
    steps = 40
    for i in range(steps):
        ratio = i / (steps - 1)
        color = _interpolate_color(ratio)
        y0 = bar_y + (steps - i - 1) * (bar_height / steps)
        elements.append(
            f'<rect x="{bar_x}" y="{y0}" width="{bar_width}" height="{bar_height / steps}" fill="{color}" stroke="none" />'
        )
    elements.append(
        f'<rect x="{bar_x}" y="{bar_y}" width="{bar_width}" height="{bar_height}" fill="none" stroke="#333" stroke-width="1" />'
    )
    elements.append(
        f'<text x="{bar_x + bar_width / 2:.1f}" y="{bar_y - 8}" font-family="sans-serif" font-size="12" text-anchor="middle">{colorbar_label}</text>'
    )
    elements.append(
        f'<text x="{bar_x + bar_width + 10}" y="{bar_y + 4}" font-family="sans-serif" font-size="12">{_format_linear(data_max)}</text>'
    )
    elements.append(
        f'<text x="{bar_x + bar_width + 10}" y="{bar_y + bar_height}" font-family="sans-serif" font-size="12">{_format_linear(data_min)}</text>'
    )

    svg_content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        *elements,
        "</svg>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg_content), encoding="utf-8")
