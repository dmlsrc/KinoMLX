"""Both model CLIs resolve prompt files with ordinary config precedence."""

import importlib

import pytest

from kinomlx.config import ConfigError


@pytest.fixture(params=["ltx2", "krea2"])
def assemble_prompt(request):
    model = request.param
    arguments = importlib.import_module(f"kinomlx.models.{model}.arguments")
    config = importlib.import_module(f"kinomlx.models.{model}.config")

    def assemble(*args):
        return config.assemble(arguments.build_parser().parse_args(list(args)))

    return model, assemble


def prompt_text(invocation):
    value = invocation.request.prompt
    return value[0] if isinstance(value, tuple) else value


def test_file_is_one_multiline_prompt_and_effective_config_contains_text(assemble_prompt, tmp_path):
    _, assemble = assemble_prompt
    path = tmp_path / "prompt with spaces.txt"
    path.write_text("\ufeff  First line.\nSecond line.\n", encoding="utf-8")
    invocation = assemble("--prompt-file", str(path))
    assert prompt_text(invocation) == "First line.\nSecond line."
    effective = getattr(invocation, "resolved_config", None)
    if effective is None:
        effective = invocation.resolved
    assert "prompt_file" not in effective["generate"]
    assert effective["generate"]["prompt"] in (
        "First line.\nSecond line.",
        ("First line.\nSecond line.",),
    )


@pytest.mark.parametrize("contents", [b"", b"  \n", b"bad\xff", b"bad\x00text"])
def test_invalid_file_is_an_actionable_error(assemble_prompt, tmp_path, contents):
    _, assemble = assemble_prompt
    path = tmp_path / "invalid.txt"
    path.write_bytes(contents)
    with pytest.raises(ConfigError, match="prompt file"):
        assemble("--prompt-file", str(path))


def test_missing_file_and_conflicting_inputs_fail(assemble_prompt, tmp_path):
    _, assemble = assemble_prompt
    with pytest.raises(ConfigError, match="not a file"):
        assemble("--prompt-file", str(tmp_path / "missing.txt"))
    path = tmp_path / "prompt.txt"
    path.write_text("from file")
    with pytest.raises(ConfigError, match="not both"):
        assemble("--prompt", "literal", "--prompt-file", str(path))


def test_toml_cli_and_set_precedence(assemble_prompt, tmp_path):
    model, assemble = assemble_prompt
    path = tmp_path / "prompt.txt"
    path.write_text("from file")
    config = tmp_path / "config.toml"
    config.write_text(f'model = "{model}"\n[generate]\nprompt_file = "{path}"\n')
    assert prompt_text(assemble("--config", str(config))) == "from file"
    assert prompt_text(assemble("--config", str(config), "--prompt", "CLI")) == "CLI"
    config.write_text(f'model = "{model}"\n[generate]\nprompt = "TOML"\n')
    assert prompt_text(assemble("--config", str(config), "--prompt-file", str(path))) == "from file"
    assert (
        prompt_text(assemble("--config", str(config), "--set", f"generate.prompt_file={path}"))
        == "from file"
    )
