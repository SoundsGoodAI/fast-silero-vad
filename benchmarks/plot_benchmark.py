#!/usr/bin/env python3
"""Create static duration-sweep figures for README and release notes.

The plotter combines Python and C++ benchmark TSV rows. It writes one figure
with absolute RTFx and another normalized to the official Python ONNX result at
each input duration. Plotly and Kaleido remain optional runtime dependencies.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go

REQUIRED_COLUMNS = {"language", "benchmark", "audio_duration_ms", "rtfx"}


def parse_args() -> argparse.Namespace:
    """Parse benchmark plotting command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed plotting options.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Create static Plotly plots from fast-silero-vad benchmark TSV files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-tsv-path",
        required=True,
        nargs="+",
        type=str,
        help="Benchmark TSV files written by the benchmark commands.",
    )
    parser.add_argument(
        "--output-dir-path",
        required=True,
        type=str,
        help="Directory where static plot files should be written.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["svg"],
        choices=("svg", "png"),
        help="Static output formats to write.",
    )
    return parser.parse_args()


def read_rows(paths: list[str | Path]) -> list[dict[str, str]]:
    """Read benchmark rows from one or more TSV files.

    Parameters
    ----------
    paths : list[str | Path]
        Input benchmark TSV paths.

    Returns
    -------
    list[dict[str, str]]
        Benchmark rows preserving file order.

    Raises
    ------
    ValueError
        If an input TSV is missing a required duration-sweep column or all
        input files are empty.
    """

    rows = []
    for path in paths:
        with open(path, encoding="utf8", newline="") as input_file:
            reader = csv.DictReader(input_file, delimiter="\t")
            missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or ())
            if missing_columns:
                raise ValueError(
                    f"Benchmark TSV {path} is missing columns "
                    f"{sorted(missing_columns)}."
                )
            rows.extend(reader)
    if not rows:
        raise ValueError("Benchmark TSV files contain no measurement rows.")
    return rows


def create_duration_figure(rows: list[dict[str, str]], metric: str) -> go.Figure:
    """Create one combined Python and official C++ duration-sweep figure.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Validated duration-sweep rows.
    metric : str
        Metric to plot, either ``rtfx`` or ``speedup``.

    Returns
    -------
    plotly.graph_objects.Figure
        Configured duration-sweep line figure.

    Raises
    ------
    RuntimeError
        If Plotly is not installed.
    ValueError
        If the metric is unsupported, benchmark values are malformed, no
        supported implementation is present, or a Python baseline is missing.
    """

    if metric not in {"rtfx", "speedup"}:
        raise ValueError(f"Unsupported duration-sweep metric: {metric}.")

    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires optional dependencies. Install them with "
            "`uv sync --extra plot` or `pip install 'fast-silero-vad[plot]'`."
        ) from error

    curves = (
        (
            "python",
            "official_silero_onnx",
            "Silero ONNX (Python)",
            "saddlebrown",
            None,
        ),
        ("python", "fast_silero_standard", "Fast ONNX (Python)", "royalblue", None),
        (
            "python",
            "fast_silero_custom_op",
            "Fast ONNX RFFT (Python)",
            "crimson",
            None,
        ),
        ("cpp", "official_silero_onnx", "Silero ONNX (C++)", "forestgreen", None),
        (
            "python",
            "fast_silero_standard_segmenter",
            "Fast ONNX + Seg (Python)",
            "royalblue",
            "dash",
        ),
        (
            "python",
            "fast_silero_custom_op_segmenter",
            "Fast ONNX RFFT + Seg (Python)",
            "crimson",
            "dash",
        ),
    )
    python_baselines = {
        row["audio_duration_ms"]: float(row["rtfx"])
        for row in rows
        if row.get("language") == "python"
        and row["benchmark"] == "official_silero_onnx"
    }
    if metric == "speedup" and not python_baselines:
        raise ValueError("Speedup plots require official Python ONNX baseline rows.")

    figure = go.Figure()
    for language, benchmark, label, color, dash in curves:
        benchmark_rows = sorted(
            (
                row
                for row in rows
                if row.get("language") == language and row["benchmark"] == benchmark
            ),
            key=lambda row: int(row["audio_duration_ms"]),
        )
        if not benchmark_rows:
            continue
        if metric == "speedup":
            missing_baselines = {
                row["audio_duration_ms"]
                for row in benchmark_rows
                if row["audio_duration_ms"] not in python_baselines
            }
            if missing_baselines:
                raise ValueError(
                    "Missing official Python ONNX baselines for durations "
                    f"{sorted(missing_baselines, key=int)}."
                )
        figure.add_scatter(
            x=[f"{row['audio_duration_ms']} ms" for row in benchmark_rows],
            y=[
                float(row["rtfx"])
                if metric == "rtfx"
                else float(row["rtfx"]) / python_baselines[row["audio_duration_ms"]]
                for row in benchmark_rows
            ],
            mode="lines+markers",
            name=label,
            line={"color": color, "width": 5, "dash": dash},
            marker={
                "size": 15 if dash else 13,
                "symbol": "star"
                if dash
                else "diamond"
                if language == "cpp"
                else "circle",
            },
        )

    if not figure.data:
        raise ValueError("Benchmark rows do not contain a supported implementation.")

    metric_label = "RTFx" if metric == "rtfx" else "Speedup vs official Python ONNX"
    figure.update_layout(
        template="plotly_white",
        width=1100,
        height=760,
        margin={"l": 115, "r": 40, "t": 155, "b": 100},
        font={"family": "Arial", "size": 22, "weight": 700},
        xaxis={
            "title": {
                "text": "Input audio duration",
                "font": {"size": 23, "weight": 700},
            },
            "tickfont": {"size": 21, "weight": 700},
        },
        yaxis={
            "title": {"text": metric_label, "font": {"size": 23, "weight": 700}},
            "tickfont": {"size": 21, "weight": 700},
            "rangemode": "tozero",
            "gridcolor": "gainsboro",
        },
        legend={
            "orientation": "h",
            "y": 1.28,
            "x": 0.5,
            "xanchor": "center",
            "entrywidth": 270,
            "entrywidthmode": "pixels",
            "font": {"size": 16, "weight": 700},
            "traceorder": "normal",
        },
    )
    return figure


def run(args: argparse.Namespace) -> None:
    """Read benchmark rows and write static Plotly plots.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed input paths, output directory, and static output formats.

    Raises
    ------
    RuntimeError
        If plotting or static rendering dependencies are unavailable.
    ValueError
        If input benchmark data is empty, malformed, or incomplete.
    """

    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input_tsv_path)

    for metric in ("rtfx", "speedup"):
        figure = create_duration_figure(rows, metric)
        for output_format in args.formats:
            try:
                figure.write_image(
                    f"{output_dir}/duration_sweep_combined_{metric}.{output_format}"
                )
            except (RuntimeError, ValueError) as error:
                raise RuntimeError(
                    "Static Plotly export requires Kaleido. Install plotting extras "
                    "with `uv sync --extra plot`. Kaleido v1 also requires Chrome; "
                    "install it with `uv run --extra plot plotly_get_chrome -y` when "
                    "needed."
                ) from error


def main() -> None:
    """Parse CLI arguments and write benchmark plots."""

    run(parse_args())


if __name__ == "__main__":
    main()
