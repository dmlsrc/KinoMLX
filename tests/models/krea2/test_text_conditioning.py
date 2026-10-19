"""Krea text stage lifetime, slicing, and provenance contracts."""

import weakref
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest

import kinomlx.models.krea2.text_conditioning as stage_module
from kinomlx.config_rules import validate_record
from kinomlx.models.krea2.pipelines.text_to_image import _negative_prompts
from kinomlx.models.krea2.request_sources import resolve_request_sources
from kinomlx.models.krea2.resources import prepare_resources
from kinomlx.models.krea2.settings import Krea2Settings
from kinomlx.models.krea2.source_loading import load_component_tokenizer
from kinomlx.models.krea2.text_conditioning import (
    Krea2PromptConditioning,
    Krea2TextConditioner,
    Krea2TextConditioning,
    load_text_conditioning,
    read_text_conditioning_prompts,
    text_conditioning_artifact,
    text_conditioning_provenance,
)
from kinomlx.models.krea2.text_encoder import Qwen3VLLoadReceipt
from kinomlx.models.krea2.tokenizer import Krea2TokenBatch
from kinomlx.models.krea2.types import Krea2Request
from kinomlx.settings import Settings
from kinomlx.tokenizers.loading import load_tokenizer_tables

from ._fixtures import make_snapshot


def _token_batch(prompts: tuple[str, ...]) -> Krea2TokenBatch:
    rows = tuple(
        tuple(prompt_index * 1000 + token_index for token_index in range(546))
        for prompt_index, _prompt in enumerate(prompts)
    )
    masks = tuple(
        (True,) * (36 + prompt_index) + (False,) * (505 - prompt_index) + (True,) * 5
        for prompt_index, _prompt in enumerate(prompts)
    )

    def position_row(mask: tuple[bool, ...]) -> tuple[int, ...]:
        position = -1
        result: list[int] = []
        for valid in mask:
            if valid:
                position += 1
                result.append(position)
            else:
                result.append(1)
        return tuple(result)

    positions = tuple(position_row(mask) for mask in masks)
    return Krea2TokenBatch(
        input_ids=rows,
        attention_mask=masks,
        position_ids=(
            positions,
            positions,
            positions,
        ),
        main_valid_lengths=tuple(36 + index for index, _prompt in enumerate(prompts)),
        max_sequence_length=512,
        identity="tokens:" + ",".join(prompts),
    )


def test_native_stage_encodes_both_branches_then_releases_qwen(
    tmp_path,
    monkeypatch,
) -> None:
    source = make_snapshot(tmp_path / "source", variant="turbo")
    resources = prepare_resources(
        Krea2Settings(source_path=source),
        infrastructure=Settings(
            hf_home=tmp_path / "hf",
        ),
    )
    events: list[str] = []
    tokenizer_refs: list[weakref.ReferenceType] = []

    class _FakeTokenizer:
        tables = load_component_tokenizer(resources.tokenizer).tables

    def make_tokenizer(_path):
        tokenizer = _FakeTokenizer()
        tokenizer_refs.append(weakref.ref(tokenizer))
        return tokenizer

    class _FakeModel:
        def __init__(self, _config, selected_layers: tuple[int, ...]) -> None:
            assert all(reference() is None for reference in tokenizer_refs)
            self.config = _config
            self.selected_layers = selected_layers
            self.calls = 0

        def __call__(
            self,
            input_ids: mx.array,
            *,
            attention_mask: mx.array,
            position_ids: mx.array,
            reporter=None,
        ) -> SimpleNamespace:
            del attention_mask, position_ids, reporter
            self.calls += 1
            events.append(f"encode:{self.calls}")
            base = input_ids.astype(mx.float32)[:, :, None, None]
            taps = mx.arange(len(self.selected_layers), dtype=mx.float32)[None, None, :, None]
            hidden = mx.broadcast_to(
                base + taps * 10_000 + self.calls * 100_000,
                (input_ids.shape[0], input_ids.shape[1], len(self.selected_layers), 4),
            )
            return SimpleNamespace(hidden_states=hidden)

    def fake_load(_model, *, source_bindings=(), reporter=None) -> Qwen3VLLoadReceipt:
        del reporter
        events.append("load")
        count, digest = stage_module.qwen3_vl_target_identity(
            _model.config,
            _model.selected_layers,
        )
        return Qwen3VLLoadReceipt(
            target_count=count,
            target_digest=digest,
            source_files=("text_encoder-00000.safetensors",),
        )

    monkeypatch.setattr(stage_module, "Qwen3VLTextModel", _FakeModel)
    monkeypatch.setattr(stage_module, "load_qwen3_vl_text_weights", fake_load)
    monkeypatch.setattr(stage_module, "load_component_tokenizer", make_tokenizer)
    monkeypatch.setattr(
        stage_module,
        "prepare_krea2_token_batch",
        lambda _tokenizer, prompts, **_kwargs: _token_batch(prompts),
    )
    monkeypatch.setattr(stage_module, "_cleanup_qwen", lambda: events.append("cleanup"))

    result = Krea2TextConditioner()(
        ("positive one", "positive two"),
        resources,
        negative_prompts=("negative one", "negative two"),
    )

    assert events == ["load", "encode:1", "encode:2", "cleanup"]
    assert result.positive.hidden_states.shape == (2, 512, 12, 4)
    assert result.negative is not None
    assert result.negative.hidden_states.shape == (2, 512, 12, 4)
    assert result.positive.hidden_states[0, 0, 0, 0].item() == 100_034
    assert result.positive.hidden_states[1, 0, 0, 0].item() == 101_034
    assert result.positive.hidden_states[0, 0, 1, 0].item() == 110_034
    assert result.negative.hidden_states[0, 0, 0, 0].item() == 200_034
    assert result.positive.attention_mask.shape == (2, 512)
    assert result.positive.position_ids.shape == (3, 2, 512)
    expected = _token_batch(("positive one", "positive two"))
    assert mx.array_equal(
        result.positive.attention_mask,
        mx.array(expected.conditioning_attention_mask, dtype=mx.bool_),
    ).item()
    assert mx.array_equal(
        result.positive.position_ids,
        mx.array(expected.conditioning_position_ids, dtype=mx.int32),
    ).item()
    assert result.provenance.is_distilled is True
    assert result.provenance.declared_variant == "turbo"
    assert result.provenance.qwen_target_count == 386
    assert result.provenance.compatibility_identity

    raw_authority = replace(
        resources,
        capabilities=replace(resources.capabilities, is_distilled=False),
    )
    raw_result = Krea2TextConditioner()("positive", raw_authority)
    assert raw_result.provenance.is_distilled is False
    assert raw_result.negative is None
    assert events[-3:] == ["load", "encode:1", "cleanup"]


def test_native_stage_rejects_mismatched_negative_batch_before_qwen(tmp_path) -> None:
    source = make_snapshot(tmp_path / "source", variant="turbo")
    resources = prepare_resources(
        Krea2Settings(source_path=source),
        infrastructure=Settings(
            hf_home=tmp_path / "hf",
        ),
    )

    with pytest.raises(ValueError, match="batch size"):
        Krea2TextConditioner()(
            ("one", "two"),
            resources,
            negative_prompts=("only one",),
        )


def _small_resources(tmp_path):
    source = make_snapshot(tmp_path / "source", variant="turbo")
    resources = prepare_resources(
        Krea2Settings(source_path=source),
        infrastructure=Settings(
            hf_home=tmp_path / "hf",
        ),
    )
    resources = resources
    return replace(
        resources,
        text_config=replace(resources.text_config, hidden_size=4, intermediate_size=8),
    )


def _saved_conditioning(
    resources, *, negative: bool = True, prompts=("first prompt", "second prompt")
) -> Krea2TextConditioning:
    positions = mx.broadcast_to(
        mx.arange(512, dtype=mx.int32)[None, None, :],
        (3, len(prompts), 512),
    )

    def branch(prefix: str, branch_prompts: tuple[str, ...]) -> Krea2PromptConditioning:
        return Krea2PromptConditioning(
            hidden_states=mx.full(
                (len(prompts), 512, 12, 4),
                1.0 if prefix == "positive" else -1.0,
                dtype=mx.bfloat16,
            ),
            attention_mask=mx.ones((len(prompts), 512), dtype=mx.bool_),
            position_ids=positions,
            prompts=branch_prompts,
            token_identity=f"tokens:{prefix}",
        )

    return Krea2TextConditioning(
        positive=branch("positive", prompts),
        negative=(branch("negative", ("", "soft focus")) if negative else None),
        provenance=text_conditioning_provenance(resources),
    )


def _write_conditioning(path, conditioning: Krea2TextConditioning) -> None:
    artifact = text_conditioning_artifact(conditioning)
    mx.save_safetensors(
        str(path),
        dict(artifact.tensors),
        metadata=dict(artifact.metadata),
    )


@pytest.mark.parametrize("policy", ["require", "observe"])
def test_saved_custom_system_prompt_roundtrip_and_mismatch(tmp_path, monkeypatch, policy):
    resources = _small_resources(tmp_path)
    monkeypatch.setattr(
        stage_module,
        "load_component_tokenizer",
        lambda path: SimpleNamespace(encode=lambda text: tuple(range(80))),
    )
    saved = _saved_conditioning(resources)
    provenance = text_conditioning_provenance(
        resources, system_prompt="Describe the visible scene.", prefix_tokens=80
    )
    saved = replace(
        saved,
        provenance=provenance,
        positive=replace(saved.positive, position_ids=saved.positive.position_ids + 80),
        negative=replace(saved.negative, position_ids=saved.negative.position_ids + 80),
    )
    path = tmp_path / "custom.safetensors"
    _write_conditioning(path, saved)
    loaded = load_text_conditioning(
        path, resources, system_prompt="Describe the visible scene.", metadata_policy=policy
    )
    assert loaded.provenance == provenance
    assert mx.array_equal(loaded.positive.hidden_states, saved.positive.hidden_states).item()
    for override in (None, "", "A different instruction"):
        with pytest.raises(ValueError, match="system prompt does not match"):
            load_text_conditioning(path, resources, system_prompt=override, metadata_policy=policy)


def test_system_prompt_request_rejects_edit_noop_and_keeps_empty():
    assert Krea2Request(prompt="scene", system_prompt="").system_prompt == ""
    with pytest.raises(ValueError, match="grounding_system_prompt"):
        validate_record(
            Krea2Request(prompt="scene", source_images=("reference.png",), system_prompt=""),
            location="[generate]",
        )
    with pytest.raises(ValueError, match="system_prompt.*at most 65536"):
        Krea2Request(prompt="scene", system_prompt="x" * 65_537)


def test_system_prompt_cli_and_toml_preserve_empty_override(tmp_path):
    from kinomlx.models.krea2.arguments import build_parser
    from kinomlx.models.krea2.config import assemble

    config = tmp_path / "instruction.toml"
    config.write_text(
        '[generate]\nprompt = "A landscape."\nsystem_prompt = "Describe the lighting."\n'
    )
    loaded = assemble(build_parser().parse_args(["--config", str(config)]))
    assert loaded.request.system_prompt == "Describe the lighting."
    overridden = assemble(
        build_parser().parse_args(["--config", str(config), "--system-prompt", ""])
    )
    assert overridden.request.system_prompt == ""
    assert overridden.resolved_config["generate"]["system_prompt"] == ""


def test_saved_text_conditioning_round_trips_and_supplies_request_prompts(tmp_path) -> None:
    resources = _small_resources(tmp_path)
    expected = _saved_conditioning(resources)
    sidecar = tmp_path / "prompt_text.safetensors"
    _write_conditioning(sidecar, expected)

    prompts, negative = read_text_conditioning_prompts(sidecar)
    request = resolve_request_sources(Krea2Request(text_conditioning=sidecar))
    loaded = load_text_conditioning(
        sidecar,
        resources,
        expected_prompts=request.expanded_prompts(),
        expected_negative_prompts=request.expanded_negative_prompts(),
    )

    assert prompts == expected.positive.prompts
    assert expected.negative is not None
    assert negative == expected.negative.prompts
    assert request.prompts == expected.positive.prompts
    assert request.negative_prompts == expected.negative.prompts
    assert loaded.provenance == expected.provenance
    assert mx.array_equal(
        loaded.positive.hidden_states,
        expected.positive.hidden_states,
    ).item()
    assert loaded.negative is not None
    assert mx.array_equal(
        loaded.negative.hidden_states,
        expected.negative.hidden_states,
    ).item()


def test_replay_preserves_negative_prompt_provenance_through_request_replacement(tmp_path) -> None:
    resources = _small_resources(tmp_path)
    sidecar = tmp_path / "prompt_text.safetensors"
    saved = _saved_conditioning(resources)
    _write_conditioning(sidecar, saved)
    request = resolve_request_sources(Krea2Request(text_conditioning=sidecar))

    for inherited in (
        request,
        resolve_request_sources(replace(request, seed=99)),
        resolve_request_sources(replace(request, text_conditioning=sidecar)),
    ):
        assert inherited.negative_prompt is None
        assert inherited.negative_prompts == saved.negative.prompts
        assert _negative_prompts(inherited, cfg_enabled=False) is None
        assert _negative_prompts(inherited, cfg_enabled=True) == saved.negative.prompts

    explicit = resolve_request_sources(replace(request, negative_prompt="explicit negative"))
    with pytest.raises(ValueError, match="negative_prompt is unused"):
        _negative_prompts(explicit, cfg_enabled=False)
    assert _negative_prompts(explicit, cfg_enabled=True) == ("explicit negative",) * 2


@pytest.mark.parametrize("policy", ["require", "observe"])
def test_replay_batch_mismatch_reports_saved_and_expanded_counts(
    tmp_path, monkeypatch, policy
) -> None:
    resources = _small_resources(tmp_path)
    sidecar = tmp_path / "prompt_text.safetensors"
    _write_conditioning(sidecar, _saved_conditioning(resources, negative=False))
    request = resolve_request_sources(Krea2Request(text_conditioning=sidecar, images_per_prompt=2))
    monkeypatch.setattr(
        stage_module, "load_weights", lambda _path: pytest.fail("batch mismatch loaded tensors")
    )

    with pytest.raises(ValueError, match="has 2 rows.*expands to 4 rows.*images_per_prompt=1"):
        load_text_conditioning(
            sidecar, resources, expected_prompts=request.expanded_prompts(), metadata_policy=policy
        )


def test_saved_negative_rows_are_not_expanded_twice(tmp_path) -> None:
    resources = _small_resources(tmp_path)
    saved = _saved_conditioning(resources)
    assert saved.negative is not None
    saved = replace(
        saved,
        positive=replace(saved.positive, prompts=("fixture prompt",) * 2),
        negative=replace(saved.negative, prompts=("saved negative",) * 2),
    )
    sidecar = tmp_path / "prompt_text.safetensors"
    _write_conditioning(sidecar, saved)
    request = resolve_request_sources(
        Krea2Request(prompt="fixture prompt", images_per_prompt=2, text_conditioning=sidecar)
    )

    assert request.expanded_negative_prompts() == ("saved negative",) * 2
    loaded = load_text_conditioning(
        sidecar,
        resources,
        expected_prompts=request.expanded_prompts(),
        expected_negative_prompts=request.expanded_negative_prompts(),
    )
    assert loaded.negative is not None
    assert loaded.negative.prompts == ("saved negative",) * 2


def test_saved_text_replay_requires_identity_but_restart_can_observe_it(tmp_path) -> None:
    resources = _small_resources(tmp_path)
    sidecar = tmp_path / "prompt_text.safetensors"
    _write_conditioning(sidecar, _saved_conditioning(resources, negative=False))
    changed = replace(
        resources,
        text_encoder=replace(resources.text_encoder, graph_fingerprint="different-text-graph"),
    )

    with pytest.raises(ValueError, match="provenance does not match"):
        load_text_conditioning(sidecar, changed)

    loaded = load_text_conditioning(sidecar, changed, metadata_policy="observe")
    assert loaded.replay_receipt is not None
    assert loaded.replay_receipt["identity_match"] is False

    recipe_only = replace(resources, recipe_identity="different-recipe")
    recipe_replay = load_text_conditioning(sidecar, recipe_only)
    assert recipe_replay.replay_receipt is not None
    assert recipe_replay.replay_receipt["identity_match"] is True
    assert (
        recipe_replay.provenance.compatibility_identity
        == text_conditioning_provenance(recipe_only).compatibility_identity
    )


def test_saved_text_shape_is_rejected_before_payload_materialization(
    tmp_path,
    monkeypatch,
) -> None:
    resources = _small_resources(tmp_path)
    expected = _saved_conditioning(resources, negative=False)
    artifact = text_conditioning_artifact(expected)
    tensors = dict(artifact.tensors)
    tensors["positive_hidden_states"] = mx.zeros(
        (2, 511, 12, 4),
        dtype=mx.bfloat16,
    )
    sidecar = tmp_path / "bad_text.safetensors"
    mx.save_safetensors(str(sidecar), tensors, metadata=dict(artifact.metadata))
    monkeypatch.setattr(
        stage_module,
        "load_weights",
        lambda _path: pytest.fail("invalid header reached payload materialization"),
    )

    with pytest.raises(ValueError, match="must have shape"):
        load_text_conditioning(sidecar, resources)


def test_empty_saved_user_prompt_roundtrip_and_conflict(tmp_path):
    resources = _small_resources(tmp_path)
    stage = _saved_conditioning(resources, negative=False, prompts=("",))
    path = tmp_path / "empty.safetensors"
    _write_conditioning(path, stage)
    for explicit in (False, True):
        request = resolve_request_sources(
            Krea2Request(text_conditioning=path, prompt_unconditional=explicit)
        )
        assert request.prompts == ("",)
        loaded = load_text_conditioning(
            path,
            resources,
            expected_prompts=request.prompts,
            expected_negative_prompts=None,
        )
        assert loaded.positive.prompts == ("",)
    _write_conditioning(path, _saved_conditioning(resources, negative=False))
    with pytest.raises(ValueError, match="empty saved"):
        resolve_request_sources(Krea2Request(text_conditioning=path, prompt_unconditional=True))


def test_native_stage_reuses_loaded_tables_for_identity(tmp_path, monkeypatch) -> None:
    resources = _small_resources(tmp_path)
    derivations: list[object] = []

    def counted_tables(*args, **kwargs):
        derivations.append(args)
        return load_tokenizer_tables(*args, **kwargs)

    monkeypatch.setattr("kinomlx.models.krea2.resource_types.load_tokenizer_tables", counted_tables)
    loads: list[object] = []

    class _FakeModel:
        def __init__(self, config, selected_layers: tuple[int, ...]) -> None:
            self.config = config
            self.selected_layers = selected_layers

        def __call__(self, input_ids, *, attention_mask, position_ids, reporter=None):
            del attention_mask, position_ids, reporter
            hidden = mx.zeros(
                (input_ids.shape[0], input_ids.shape[1], len(self.selected_layers), 4)
            )
            return SimpleNamespace(hidden_states=hidden)

    def fake_load(model, *, source_bindings=(), reporter=None) -> Qwen3VLLoadReceipt:
        del reporter
        loads.append(source_bindings)
        count, digest = stage_module.qwen3_vl_target_identity(model.config, model.selected_layers)
        return Qwen3VLLoadReceipt(count, digest, ("model.safetensors",))

    monkeypatch.setattr(stage_module, "Qwen3VLTextModel", _FakeModel)
    monkeypatch.setattr(stage_module, "load_qwen3_vl_text_weights", fake_load)
    monkeypatch.setattr(
        stage_module,
        "prepare_krea2_token_batch",
        lambda _tokenizer, prompts, **_kwargs: _token_batch(prompts),
    )
    monkeypatch.setattr(stage_module, "_cleanup_qwen", lambda: None)

    result = Krea2TextConditioner()(("one prompt",), resources)

    assert loads == [resources.text_encoder.bindings]
    assert len(derivations) == 1
    assert result.provenance.is_compatible_with(text_conditioning_provenance(resources))
