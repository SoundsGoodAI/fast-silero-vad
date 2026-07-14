"""Shared pytest command-line options."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--download-silero-vad",
        action="store_true",
        help="Use the silero-vad package checkpoint for integration tests.",
    )
