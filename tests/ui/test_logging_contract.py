"""Static contract: package modules and developer tools use logging."""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def test_source_uses_logging_instead_of_direct_terminal_output(parsed_project_sources) -> None:
    offenders: list[str] = []
    for path, tree in parsed_project_sources.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _dotted_name(node.func)
            direct = name == "print" or name in {
                "sys.stdout.write",
                "sys.stderr.write",
            }
            console_write = name is not None and name.endswith(".print")
            if direct or console_write:
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}: {name}")
    assert offenders == []


def test_console_rows_end_at_their_text() -> None:
    import io
    import logging

    from rich.console import Console

    from kinomlx.ui.logging import _ScrollbackFriendlyHandler

    buffer = io.StringIO()
    console = Console(file=buffer, width=80, force_terminal=False, color_system=None)
    handler = _ScrollbackFriendlyHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=False,
        log_time_format="[%H:%M:%S]",
    )
    logger = logging.getLogger("kinomlx.test.unpadded_rows")
    logger.handlers[:] = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.info("Run:")
    logger.info(
        "  Model: LTX-2.3 | Action: generate | Resolution: 768x448 | Frames: 121 | FPS: 24 | Seed: 42"
    )

    lines = buffer.getvalue().splitlines()
    assert lines
    assert all(line == line.rstrip() for line in lines)
    assert lines[0].endswith("Run:")
    assert any("Model: LTX-2.3" in line for line in lines)
    assert any("Seed: 42" in line for line in lines)


def test_configuring_quiet_logging_does_not_outlive_its_scope() -> None:
    import logging

    from kinomlx.ui.logging import configure_logging
    from tests._logging import PACKAGE_LOGGER, package_logger_state

    logger = logging.getLogger(PACKAGE_LOGGER)
    before = (logger.level, list(logger.handlers), logger.propagate)

    with package_logger_state():
        configure_logging(quiet=True)
        assert logger.level == logging.WARNING
        assert logger.handlers != before[1]

    assert (logger.level, list(logger.handlers), logger.propagate) == before
