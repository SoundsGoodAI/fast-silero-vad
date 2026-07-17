#!/usr/bin/env python3
"""Build the benchmark around Silero VAD's upstream C++ ONNX example.

The builder downloads a pinned copy of the upstream C++ implementation and
headers matching the installed ONNX Runtime. It then compiles the local timing
harness and packages the runtime shared library beside the executable.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import onnxruntime

from fast_silero_vad.constants import VAD_ORT_HEADER_PATHS
from fast_silero_vad.export.custom_op.build_vad_custom_op import (
    download_onnxruntime_headers,
    download_url,
)

SILERO_VAD_REVISION = "b163605b3f44c3aadf28f97b125a2f7c461e9a7f"
SILERO_VAD_CPP_FILES = ("silero.cc", "silero.h", "wav.h")


def download_silero_vad_cpp(source_dir: Path) -> None:
    """Download missing files from the pinned upstream C++ implementation.

    Files are downloaded through a temporary directory so interrupted requests
    cannot leave partial source files in the persistent cache.

    Parameters
    ----------
    source_dir : Path
        Cache directory for the pinned upstream C++ source files.

    Raises
    ------
    OSError
        If a source file cannot be downloaded or written.
    """

    source_dir.mkdir(parents=True, exist_ok=True)
    missing_files = [
        filename
        for filename in SILERO_VAD_CPP_FILES
        if not (source_dir / filename).is_file()
    ]
    if not missing_files:
        return

    with tempfile.TemporaryDirectory(dir=source_dir) as temporary_dir:
        temporary_source_dir = Path(temporary_dir)
        for filename in missing_files:
            url = (
                "https://raw.githubusercontent.com/snakers4/silero-vad/"
                f"{SILERO_VAD_REVISION}/examples/c%2B%2B/{filename}"
            )
            download_url(url, temporary_source_dir / filename)
        for filename in missing_files:
            shutil.copy2(temporary_source_dir / filename, source_dir / filename)


def get_onnxruntime_library(runtime_dir: Path, version: str) -> tuple[Path, str, str]:
    """Return the ONNX Runtime library and local dynamic-linker metadata.

    Parameters
    ----------
    runtime_dir : Path
        ONNX Runtime package directory containing native libraries.
    version : str
        Installed ONNX Runtime version.

    Returns
    -------
    tuple[Path, str, str]
        Versioned runtime library, required local link name, and executable
        runtime search path.

    Raises
    ------
    RuntimeError
        If the platform is unsupported or the expected library is absent.
    """

    major_version = version.partition(".")[0]
    if sys.platform.startswith("linux"):
        library_name = f"libonnxruntime.so.{version}"
        link_name = f"libonnxruntime.so.{major_version}"
        runtime_search_path = "$ORIGIN/lib"
    elif sys.platform == "darwin":
        library_name = f"libonnxruntime.{version}.dylib"
        link_name = f"libonnxruntime.{major_version}.dylib"
        runtime_search_path = "@loader_path/lib"
    else:
        raise RuntimeError("The C++ benchmark supports only Linux and macOS.")

    runtime_library = runtime_dir / library_name
    if not runtime_library.is_file():
        raise RuntimeError(f"Cannot find {runtime_library}.")
    return runtime_library, link_name, runtime_search_path


def parse_args() -> argparse.Namespace:
    """Parse C++ benchmark build arguments.

    Returns
    -------
    argparse.Namespace
        Parsed executable output path.
    """

    parser = argparse.ArgumentParser(
        description="Build the official Silero VAD C++ ONNX benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="benchmarks/cpp/build/vad_benchmark",
        help="Compiled benchmark executable.",
    )
    return parser.parse_args()


def main() -> None:
    """Download dependencies and compile the C++ benchmark executable.

    Raises
    ------
    RuntimeError
        If the platform, compiler, or ONNX Runtime installation is unsupported.
    OSError
        If source files, headers, or runtime libraries cannot be copied.
    subprocess.CalledProcessError
        If the C++ compiler exits unsuccessfully.
    """

    args = parse_args()
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("Cannot find a C++ compiler named `c++`.")

    source_path = Path(__file__).with_name("benchmark.cc")
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    version = onnxruntime.__version__
    cache_root = Path.home() / ".cache" / "fast-silero-vad"
    include_root = cache_root / f"onnxruntime-{version}" / "include"
    if not all(
        (include_root / header_path).exists() for header_path in VAD_ORT_HEADER_PATHS
    ):
        include_root.mkdir(parents=True, exist_ok=True)
        download_onnxruntime_headers(include_root, version)

    source_dir = cache_root / f"silero-vad-{SILERO_VAD_REVISION}" / "examples-cpp"
    download_silero_vad_cpp(source_dir)

    runtime_dir = Path(onnxruntime.__file__).resolve().parent / "capi"
    runtime_library, link_name, runtime_search_path = get_onnxruntime_library(
        runtime_dir, version
    )
    library_dir = output_path.parent / "lib"
    library_dir.mkdir(parents=True, exist_ok=True)
    packaged_runtime_library = library_dir / runtime_library.name
    shutil.copy2(runtime_library, packaged_runtime_library)

    runtime_link = library_dir / link_name
    runtime_link.unlink(missing_ok=True)
    runtime_link.symlink_to(packaged_runtime_library.name)

    with tempfile.TemporaryDirectory(
        prefix="fast-silero-vad-cpp-build-", dir=output_path.parent
    ) as build_dir:
        temporary_output_path = Path(build_dir) / output_path.name
        command = [
            compiler,
            "-std=c++20",
            "-O3",
            "-DNDEBUG",
            "-Wno-unused-result",
            "-DUSE_ONNX=1",
            str(source_path),
            str(source_dir / "silero.cc"),
            "-I",
            str(source_dir),
            "-I",
            str(include_root),
            "-I",
            str(include_root / "onnxruntime/core/session"),
            str(packaged_runtime_library),
            f"-Wl,-rpath,{runtime_search_path}",
            "-o",
            str(temporary_output_path),
        ]
        subprocess.run(command, check=True)
        shutil.copy2(temporary_output_path, output_path)


if __name__ == "__main__":
    main()
