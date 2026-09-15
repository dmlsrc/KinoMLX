"""Image-grounded Qwen conditioning with bounded vision and language ownership."""

import gc
from dataclasses import dataclass, replace
from typing import cast

import mlx.core as mx

from kinomlx.io.identity import payload_digest
from kinomlx.reporting import NullReporter, Reporter, reported_phase
from kinomlx.tokenizers.byte_bpe import QwenByteBPETokenizer

from .image_processing import grounding_patches
from .resource_types import Krea2Resources
from .source_loading import load_component_tokenizer
from .text_conditioning import (
    Krea2PromptConditioning,
    Krea2TextConditioning,
    PromptInput,
    text_conditioning_provenance,
)
from .text_encoder import Qwen3VLTextModel, load_qwen3_vl_text_weights
from .text_encoder.multimodal import Qwen3VLImageSpan, multimodal_positions
from .text_encoder.vision import Qwen3VLVisionConfig, Qwen3VLVisionModel, Qwen3VLVisualFeatures
from .text_encoder.vision_loading import load_qwen3_vl_vision_weights
from .tokenizer.template import KREA_MAX_PROMPT_CHARACTERS, KREA_PREFIX, KREA_SUFFIX


def grounded_tokens(
    tokenizer: QwenByteBPETokenizer,
    prompt: str,
    features: tuple[Qwen3VLVisualFeatures, ...],
    *,
    system_prompt: str = "",
) -> tuple[mx.array, tuple[Qwen3VLImageSpan, ...], int]:
    """Build an unpadded grounded user turn and record the removable prefix."""
    if len(prompt) > KREA_MAX_PROMPT_CHARACTERS:
        raise ValueError("Krea grounded prompt exceeds the character limit")
    system_prompt = system_prompt.strip()
    if len(system_prompt) > KREA_MAX_PROMPT_CHARACTERS:
        raise ValueError("grounding system prompt exceeds the character limit")
    prefix = (
        "<|im_start|>system\n" + system_prompt + "<|im_end|>\n<|im_start|>user\n"
        if system_prompt
        else KREA_PREFIX
    )
    tokens = list(tokenizer.encode(prefix))
    prefix_length = len(tokens)
    spans: list[Qwen3VLImageSpan] = []
    for feature in features:
        tokens.extend(tokenizer.encode("<|vision_start|>"))
        spans.append(Qwen3VLImageSpan(len(tokens), feature))
        tokens.extend([151655] * feature.embeddings.shape[0])
        tokens.extend(tokenizer.encode("<|vision_end|>"))
    tokens.extend(tokenizer.encode(prompt + KREA_SUFFIX))
    return mx.array([tokens], dtype=mx.int32), tuple(spans), prefix_length


def encode_grounded_prompt(
    model: Qwen3VLTextModel,
    tokenizer: QwenByteBPETokenizer,
    prompt: str,
    features: tuple[Qwen3VLVisualFeatures, ...],
    *,
    reporter: Reporter,
    system_prompt: str = "",
) -> Krea2PromptConditioning:
    ids, spans, prefix = grounded_tokens(tokenizer, prompt, features, system_prompt=system_prompt)
    positions = multimodal_positions(ids.shape[1], spans)
    mask = mx.ones(ids.shape, dtype=mx.bool_)
    output = model(
        ids, attention_mask=mask, position_ids=positions, image_spans=spans, reporter=reporter
    )
    hidden = output.hidden_states[:, prefix:]
    mx.eval(hidden)
    return Krea2PromptConditioning(
        hidden_states=hidden,
        attention_mask=mask[:, prefix:],
        position_ids=positions[:, :, prefix:],
        prompts=(prompt,),
        token_identity=payload_digest(
            {"tokens": cast(list[list[int]], ids.tolist()), "policy": "image-grounded-v1"}
        ),
    )


@dataclass(frozen=True)
class GroundedKrea2TextConditioner:
    """One or two image references; no adapter name, weights, or recognition policy."""

    images: tuple[mx.array, ...]
    grounding_px: int = 768
    system_prompt: str = ""

    def __call__(
        self,
        prompts: PromptInput,
        resources: Krea2Resources,
        *,
        negative_prompts: PromptInput | None = None,
        # Grounded turns are unpadded: the width follows the vision spans and the
        # actual prompt tokens, so the shared conditioner port's padded-width
        # limit has nothing to apply to. KREA_MAX_PROMPT_CHARACTERS bounds the
        # text instead.
        max_sequence_length: int = 512,  # noqa: ARG002
        system_prompt: str | None = None,
        reporter: Reporter | None = None,
    ) -> Krea2TextConditioning:
        if system_prompt is not None:
            raise ValueError("Use grounding_system_prompt for image editing")
        positive = (prompts,) if isinstance(prompts, str) else tuple(prompts)
        negative = (
            None
            if negative_prompts is None
            else (
                (negative_prompts,)
                if isinstance(negative_prompts, str)
                else tuple(negative_prompts)
            )
        )
        if len(positive) != 1 or (negative is not None and len(negative) != 1):
            raise ValueError("Krea editing currently requires one output image per call")
        sink = reporter if reporter is not None else NullReporter()
        features: list[Qwen3VLVisualFeatures] = []
        vision = Qwen3VLVisionModel(Qwen3VLVisionConfig())
        try:
            with reported_phase(sink, "load vision encoder"):
                files = tuple(
                    sorted({binding.source_file for binding in resources.text_encoder.bindings})
                )
                if len(files) != 1:
                    raise ValueError(
                        "grounded Qwen requires the complete single-file source checkpoint"
                    )
                load_qwen3_vl_vision_weights(vision, files[0], dtype=mx.bfloat16)
            with reported_phase(sink, "encode reference vision"):
                for pixels in self.images:
                    patches, grid = grounding_patches(pixels, longest_side=self.grounding_px)
                    features.append(vision(patches, grid=grid))
        finally:
            del vision
            gc.collect()
            mx.clear_cache()
        tokenizer = load_component_tokenizer(resources.tokenizer)
        tokenizer_identity = resources.tokenizer.tokenizer_identity
        model = Qwen3VLTextModel(resources.text_config, resources.capabilities.selected_text_layers)
        try:
            with reported_phase(sink, "load grounded language encoder"):
                receipt = load_qwen3_vl_text_weights(
                    model,
                    source_bindings=resources.text_encoder.bindings,
                )
            encoded = encode_grounded_prompt(
                model,
                tokenizer,
                positive[0],
                tuple(features),
                reporter=sink,
                system_prompt=self.system_prompt,
            )
            unconditioned = (
                None
                if negative is None
                else encode_grounded_prompt(
                    model,
                    tokenizer,
                    negative[0],
                    tuple(features),
                    reporter=sink,
                    system_prompt=self.system_prompt,
                )
            )
        finally:
            del model
            features.clear()
            gc.collect()
            mx.clear_cache()
        provenance = text_conditioning_provenance(
            resources, receipt, tokenizer_identity=tokenizer_identity
        )
        # Keep grounded contexts distinct from the existing text-only replay identity.
        provenance = replace(
            provenance, text_graph_identity=provenance.text_graph_identity + ":grounded-v1"
        )
        provenance = replace(
            provenance, compatibility_identity=payload_digest(provenance._identity_fields())
        )
        return Krea2TextConditioning(encoded, unconditioned, provenance)
