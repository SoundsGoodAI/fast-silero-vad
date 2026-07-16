"""Tests for platform-independent custom-op build preparation."""

from io import BytesIO
from pathlib import Path

import pytest

from fast_silero_vad.constants import VAD_ORT_HEADER_PATHS
from fast_silero_vad.export.custom_op import build_vad_custom_op


@pytest.mark.parametrize(
    ("platform", "library_name", "compiler_mode"),
    (
        ("linux", "vad_frontend_op.so", "-shared"),
        ("linux2", "vad_frontend_op.so", "-shared"),
        ("darwin", "vad_frontend_op.dylib", "-dynamiclib"),
    ),
)
def test_custom_op_build_uses_platform_library_name_and_compiler_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    library_name: str,
    compiler_mode: str,
) -> None:
    commands = []

    def download_headers(include_root: Path, version: str) -> None:
        for header_path in VAD_ORT_HEADER_PATHS:
            output_path = include_root / header_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(version, encoding="utf8")

    def run(cmd: list[str], check: bool) -> None:
        commands.append((cmd, check))
        Path(cmd[-1]).write_text("custom op", encoding="utf8")

    monkeypatch.setattr(build_vad_custom_op.sys, "platform", platform)
    monkeypatch.setattr(build_vad_custom_op.Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(
        build_vad_custom_op, "download_onnxruntime_headers", download_headers
    )
    monkeypatch.setattr(build_vad_custom_op.subprocess, "run", run)

    library_path = Path(build_vad_custom_op.build_vad_custom_op(tmp_path / "model"))

    assert library_path.name == library_name
    assert library_path.read_text(encoding="utf8") == "custom op"
    assert commands[0][1] is True
    assert compiler_mode in commands[0][0]


def test_custom_op_build_rejects_unsupported_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build_vad_custom_op.sys, "platform", "win32")

    with pytest.raises(RuntimeError, match="Linux and macOS"):
        build_vad_custom_op.build_vad_custom_op(tmp_path)


def test_download_onnxruntime_headers_uses_versioned_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls = []

    def download(url: str, output_path: Path) -> None:
        urls.append(url)
        output_path.write_text(url, encoding="utf8")

    monkeypatch.setattr(build_vad_custom_op, "download_url", download)
    include_root = tmp_path / "include"
    build_vad_custom_op.download_onnxruntime_headers(include_root, "1.27.0")

    assert urls == [
        (
            "https://raw.githubusercontent.com/microsoft/onnxruntime/"
            f"v1.27.0/include/{path}"
        )
        for path in VAD_ORT_HEADER_PATHS
    ]
    assert all((include_root / path).is_file() for path in VAD_ORT_HEADER_PATHS)


def test_download_url_uses_certifi_ca_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = object()
    cafiles = []
    requests = []

    def create_default_context(*, cafile: str) -> object:
        cafiles.append(cafile)
        return context

    def open_url(url: str, *, context: object) -> BytesIO:
        requests.append((url, context))
        return BytesIO(b"header")

    monkeypatch.setattr(
        build_vad_custom_op.certifi, "where", lambda: "/certifi/cacert.pem"
    )
    monkeypatch.setattr(
        build_vad_custom_op.ssl, "create_default_context", create_default_context
    )
    monkeypatch.setattr(build_vad_custom_op.request, "urlopen", open_url)

    output_path = tmp_path / "header.h"
    build_vad_custom_op.download_url("https://example.com/header.h", output_path)

    assert cafiles == ["/certifi/cacert.pem"]
    assert requests == [("https://example.com/header.h", context)]
    assert output_path.read_bytes() == b"header"
