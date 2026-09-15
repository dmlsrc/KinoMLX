"""Ordered replacement must agree across every production model assembler."""

from copy import deepcopy

import pytest

from kinomlx.config import ConfigError, SetOverrides, apply_set_overrides


def test_overlay_then_patch_retains_correct_origins():
    base = {"output": {"prefix": "old", "directory": "old-dir"}}
    original = deepcopy(base)
    overrides = SetOverrides.parse(
        ['output.prefix="discarded"', 'output={directory="new-dir"}', 'output.prefix="new"']
    )
    assert overrides.merge(base, normalized=overrides.config) == {
        "output": {"directory": "new-dir", "prefix": "new"}
    }
    assert {path: origin.ordinal for path, origin in overrides.origins.items()} == {
        ("output", "directory"): 2,
        ("output", "prefix"): 3,
    }
    assert base == original


def test_empty_group_preserves_inherited_and_prior_explicit_members():
    overrides = SetOverrides.parse(['output.prefix="discarded"', "output={}"])
    assert overrides.merge({"output": {"directory": "old"}}, normalized=overrides.config) == {
        "output": {"directory": "old", "prefix": "discarded"}
    }
    assert overrides.origins[("output", "prefix")].ordinal == 1


def test_inherited_scalar_cannot_be_traversed_even_if_later_replaced():
    overrides = SetOverrides.parse(["output.prefix=1", "output={}"])
    with pytest.raises(ConfigError, match="output is not a table"):
        overrides.merge({"output": False}, normalized=overrides.config)


def test_replacing_scalar_before_descending_is_allowed():
    overrides = SetOverrides.parse(["output={}", 'output.prefix="new"'])
    assert overrides.merge({"output": False}, normalized=overrides.config) == {
        "output": {"prefix": "new"}
    }


def test_normalization_clears_inherited_values_without_claiming_explicit_origin():
    overrides = SetOverrides.parse(["generate.seed=7"])
    assert overrides.merge(
        {"generate": {"seeds": [1, 2], "steps": 12}},
        normalized={"generate": {"seed": 7, "seeds": None}},
    ) == {"generate": {"seed": 7, "seeds": None, "steps": 12}}
    assert set(overrides.origins) == {("generate", "seed")}


def test_ordered_merge_matches_direct_assignment_without_normalization():
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2]}
    streams = (
        [],
        ["a.x=3"],
        ["a={}"],
        ["a={x=5}", "a.y=9"],
        ["a.x=5", "a={y=9}"],
        ["b=[]"],
        ["a=false", "a={}", "a.x=7"],
    )
    for stream in streams:
        overrides = SetOverrides.parse(stream)
        assert overrides.merge(base, normalized=overrides.config) == apply_set_overrides(
            base, stream
        )


@pytest.mark.parametrize("model", ["gmnet", "flux2", "krea2", "ltx2"])
@pytest.mark.parametrize("replacement", ['output={prefix="new"}', "output={}"])
def test_all_models_overlay_group_tables(tmp_path, model, replacement):
    from kinomlx.models.flux2.arguments import build_parser as flux_parser
    from kinomlx.models.flux2.config import assemble as flux_assemble
    from kinomlx.models.gmnet.arguments import build_parser as gmnet_parser
    from kinomlx.models.gmnet.config import assemble as gmnet_assemble
    from kinomlx.models.krea2.arguments import build_parser as krea_parser
    from kinomlx.models.krea2.config import assemble as krea_assemble
    from kinomlx.models.ltx2.arguments import build_parser as ltx_parser
    from kinomlx.models.ltx2.config import assemble as ltx_assemble

    parser, assemble = {
        "gmnet": (gmnet_parser, gmnet_assemble),
        "flux2": (flux_parser, flux_assemble),
        "krea2": (krea_parser, krea_assemble),
        "ltx2": (ltx_parser, ltx_assemble),
    }[model]
    config = tmp_path / "input.toml"
    config.write_text(
        f'model = "{model}"\n[output]\nprefix = "inherited-prefix"\n'
        f'directory = "{tmp_path / "inherited-directory"}"\n'
    )
    options = parser().parse_args(["--print-config", "--config", str(config), "--set", replacement])
    invocation = assemble(options)
    assert invocation.output.directory == tmp_path / "inherited-directory"
    assert invocation.output.prefix == ("new" if "new" in replacement else "inherited-prefix")


def test_group_overlay_preserves_prior_members_but_replaces_sequences():
    overrides = SetOverrides.parse(
        [
            'generate={adapter_paths=["first"],steps=8}',
            "generate={adapter_paths=[]}",
        ]
    )
    assert overrides.merge({"generate": {"guidance": 1.0}}, normalized=overrides.config) == {
        "generate": {"adapter_paths": [], "steps": 8, "guidance": 1.0}
    }
    assert overrides.origins[("generate", "steps")].ordinal == 1
    assert overrides.origins[("generate", "adapter_paths")].ordinal == 2
