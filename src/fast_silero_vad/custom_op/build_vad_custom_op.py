#!/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
"""Build the ONNX Runtime custom frontend used by optimized VAD bundles."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib import request

import onnxruntime

from fast_silero_vad.constants import VAD_ORT_HEADER_PATHS


def download_onnxruntime_headers(include_root: Path, version: str) -> None:
    """Download ONNX Runtime C/C++ headers from its versioned source tree.

    Parameters
    ----------
    include_root : Path
        Local include directory where headers should be written. The nested
        source-tree layout from ``VAD_ORT_HEADER_PATHS`` is preserved so the
        custom op can use the same include paths as ONNX Runtime.
    version : str
        ONNX Runtime version installed in the packaging environment.

    Raises
    ------
    OSError
        Raised when a header cannot be downloaded or written.
    """

    with tempfile.TemporaryDirectory(dir=include_root.parent) as tmp_dir:
        temporary_include_root = Path(tmp_dir) / "include"
        for header_path in VAD_ORT_HEADER_PATHS:
            temporary_header_path = temporary_include_root / header_path
            temporary_header_path.parent.mkdir(parents=True, exist_ok=True)
            header_url = (
                "https://raw.githubusercontent.com/microsoft/onnxruntime/"
                f"v{version}/include/{header_path}"
            )
            request.urlretrieve(header_url, temporary_header_path)

        for header_path in VAD_ORT_HEADER_PATHS:
            output_path = include_root / header_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temporary_include_root / header_path, output_path)


def build_vad_custom_op(output_dir: str | Path) -> str:
    """Build and return the packaged VAD frontend custom-op library.

    Parameters
    ----------
    output_dir : str | Path
        Model directory where the compiled shared library should be written.

    Returns
    -------
    str
        Compiled shared library path inside ``output_dir``.

    Raises
    ------
    RuntimeError
        Raised when the current platform does not support custom-op builds.
    subprocess.CalledProcessError
        Raised when the C++ compiler fails.
    """

    if sys.platform.startswith("linux"):
        library_name, compiler_mode = "vad_frontend_op.so", "-shared"
    elif sys.platform == "darwin":
        library_name, compiler_mode = "vad_frontend_op.dylib", "-dynamiclib"
    else:
        raise RuntimeError(
            "The ONNX Runtime custom frontend can currently be built only on "
            "Linux and macOS."
        )

    source_path = Path(__file__).resolve().parent / "vad_frontend.cc"
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    version = onnxruntime.__version__
    cache_root = Path.home() / ".cache" / "fast-silero-vad" / f"onnxruntime-{version}"
    include_root = cache_root / "include"
    if not all(
        (include_root / header_path).exists() for header_path in VAD_ORT_HEADER_PATHS
    ):
        include_root.mkdir(parents=True, exist_ok=True)
        download_onnxruntime_headers(include_root, version)

    library_path = output_dir_path / library_name
    with tempfile.TemporaryDirectory(prefix="fast-silero-vad-build-") as build_dir:
        temporary_library = Path(build_dir) / library_name
        cmd = [
            "c++",
            "-std=c++20",
            "-O3",
            "-DNDEBUG",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            compiler_mode,
            str(source_path),
            "-I",
            str(include_root),
            "-o",
            str(temporary_library),
        ]
        subprocess.run(cmd, check=True)
        shutil.copy2(temporary_library, library_path)

    return str(library_path)
