"""Typed metadata for the implemented LTX-2 video VAE graphs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from kinomlx.io.safetensors import read_metadata
from kinomlx.types import SpatioTemporalScaleFactors

_ENCODER_BLOCKS = frozenset(
    {
        "res_x",
        "compress_space_res",
        "compress_time_res",
        "compress_all_res",
    }
)
_DECODER_BLOCKS = frozenset(
    {
        "res_x",
        "compress_space",
        "compress_time",
        "compress_all",
    }
)


_COMPRESSION_KINDS = ("compress_all", "compress_time", "compress_space")


def is_compression_block(name: str) -> bool:
    """True for compress_* block names, with or without the _res suffix."""
    return name.removesuffix("_res") in _COMPRESSION_KINDS


def compression_strides(name: str) -> tuple[int, int, int]:
    """(time, height, width) stride triple for a compression block name."""
    kind = name.removesuffix("_res")
    if kind not in _COMPRESSION_KINDS:
        raise ValueError(f"not a compression block: {name!r}")
    time = 1 if kind == "compress_space" else 2
    space = 1 if kind == "compress_time" else 2
    return (time, space, space)


def _strict_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _strict_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _strict_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _strict_tuple(
    value: object,
    *,
    field: str,
    length: int,
) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    result = tuple(_strict_int(item, field=f"{field}[{index}]") for index, item in enumerate(value))
    if len(result) != length:
        raise ValueError(f"{field} must contain {length} values")
    return result


@dataclass(frozen=True)
class VideoVAEBlock:
    """One config-defined residual or compression block."""

    name: str
    num_layers: int | None = None
    multiplier: int = 1
    residual: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("block name must be a non-empty string")
        if self.num_layers is not None:
            _strict_int(self.num_layers, field=f"{self.name}.num_layers")
        _strict_int(self.multiplier, field=f"{self.name}.multiplier")
        _strict_bool(self.residual, field=f"{self.name}.residual")
        if self.name == "res_x" and self.num_layers is None:
            raise ValueError("res_x block requires num_layers")
        if self.name != "res_x" and self.num_layers is not None:
            raise ValueError(f"compression block {self.name!r} cannot set num_layers")

    @classmethod
    def from_raw(cls, raw: object, *, field: str) -> Self:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
            raise ValueError(f"{field} must be a two-item block specification")
        name, raw_params = raw
        if not isinstance(name, str):
            raise ValueError(f"{field} block name must be a string")
        if isinstance(raw_params, bool):
            raise ValueError(f"{field} block parameters must be an integer or table")
        if isinstance(raw_params, int):
            params: Mapping[str, object] = {"num_layers": raw_params}
        elif isinstance(raw_params, Mapping):
            params = raw_params
        else:
            raise ValueError(f"{field} block parameters must be an integer or table")

        num_layers = params.get("num_layers")
        if num_layers is not None:
            num_layers = _strict_int(num_layers, field=f"{field}.{name}.num_layers")
        multiplier = _strict_int(
            params.get("multiplier", 1),
            field=f"{field}.{name}.multiplier",
        )
        residual = _strict_bool(
            params.get("residual", False),
            field=f"{field}.{name}.residual",
        )
        if name == "res_x" and num_layers is None:
            raise ValueError(f"{field} residual block requires num_layers")
        if name != "res_x" and num_layers is not None:
            raise ValueError(f"{field} compression block {name!r} cannot set num_layers")
        return cls(
            name=name,
            num_layers=num_layers,
            multiplier=multiplier,
            residual=residual,
        )


def _parse_blocks(
    raw: object,
    *,
    field: str,
    allowed: frozenset[str],
) -> tuple[VideoVAEBlock, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    blocks = tuple(
        VideoVAEBlock.from_raw(item, field=f"{field}[{index}]") for index, item in enumerate(raw)
    )
    if not blocks:
        raise ValueError(f"{field} must not be empty")
    unsupported = sorted({block.name for block in blocks} - allowed)
    if unsupported:
        raise ValueError(
            f"{field} contains unsupported native Conv3d blocks: " + ", ".join(unsupported)
        )
    return blocks


def _scale_from_blocks(
    blocks: Sequence[VideoVAEBlock],
    *,
    patch_size: int,
) -> SpatioTemporalScaleFactors:
    time = 1
    height = patch_size
    width = patch_size
    for block in blocks:
        if block.name.startswith(("compress_time", "compress_all")):
            time *= 2
        if block.name.startswith(("compress_space", "compress_all")):
            height *= 2
            width *= 2
    return SpatioTemporalScaleFactors(time=time, height=height, width=width)


@dataclass(frozen=True)
class DiffusionVideoDecoderConfig:
    """Execution-changing fields for the LTX-2.5 diffusion decoder."""

    head_dim: int
    stage_channels: tuple[int, int, int, int, int]
    stage_depths: tuple[int, int, int, int, int]
    stage_kernels: tuple[
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, int],
    ]
    upsample_strides: tuple[
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, int],
    ]
    upsample_channel_reductions: tuple[int, int, int, int]
    stage5_kernel: tuple[int, int, int]
    patch_size: int = 4
    t_emb_dim: int = 384
    timestep_scale_multiplier: float = 1000.0
    model_output_type: Literal["x0", "v"] = "x0"
    default_num_inference_steps: int = 1
    spatial_compression_ratio: int = 32
    temporal_compression_ratio: int = 8
    inferred_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _strict_int(self.head_dim, field="decoder.head_dim")
        _strict_int(self.patch_size, field="decoder.patch_size")
        _strict_int(self.t_emb_dim, field="decoder.t_emb_dim")
        _strict_float(
            self.timestep_scale_multiplier,
            field="decoder.timestep_scale_multiplier",
        )
        _strict_int(
            self.default_num_inference_steps,
            field="decoder.default_num_inference_steps",
        )
        _strict_int(
            self.spatial_compression_ratio,
            field="decoder.spatial_compression_ratio",
        )
        _strict_int(
            self.temporal_compression_ratio,
            field="decoder.temporal_compression_ratio",
        )
        if self.model_output_type not in {"x0", "v"}:
            raise ValueError("decoder.model_output_type must be 'x0' or 'v'")
        if any(channels % self.head_dim for channels in self.stage_channels):
            raise ValueError("every diffusion decoder stage width must divide by head_dim")
        for index, reduction in enumerate(self.upsample_channel_reductions):
            expected = self.stage_channels[index] // reduction
            if self.stage_channels[index + 1] != expected:
                raise ValueError(
                    f"decoder.stage_channels[{index + 1}] must be "
                    f"stage_channels[{index}] // reduction = {expected}"
                )
        scale_t = math.prod(stride[0] for stride in self.upsample_strides)
        scale_h = math.prod(stride[1] for stride in self.upsample_strides)
        scale_w = math.prod(stride[2] for stride in self.upsample_strides)
        if scale_t != self.temporal_compression_ratio:
            raise ValueError("diffusion decoder temporal stride product does not match metadata")
        patch_scale = self.spatial_compression_ratio // self.patch_size
        if scale_h != patch_scale or scale_w != patch_scale:
            raise ValueError("diffusion decoder spatial stride product does not match metadata")

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, object],
        *,
        model_output_type: object,
    ) -> Self:
        """Parse nested ``NADiffusionDecoder`` constructor metadata."""
        class_name = raw.get("_class_name")
        if class_name != "NADiffusionDecoder":
            raise ValueError(
                f"vae.decoder._class_name must select NADiffusionDecoder, got {class_name!r}"
            )
        spatial_padding_mode = raw.get("spatial_padding_mode", "zeros")
        if spatial_padding_mode != "zeros":
            raise ValueError(
                "diffusion video decoder supports only zero spatial padding, "
                f"got {spatial_padding_mode!r}"
            )
        resampler_kind = raw.get("resampler_kind", "linear")
        if resampler_kind != "linear":
            raise ValueError(
                f"diffusion video decoder requires linear resampling, got {resampler_kind!r}"
            )
        stage_channels = _strict_tuple(
            raw.get("stage_channels"),
            field="vae.decoder.stage_channels",
            length=5,
        )
        stage_depths = _strict_tuple(
            raw.get("stage_depths"),
            field="vae.decoder.stage_depths",
            length=5,
        )
        raw_kernels = raw.get("stage_kernels")
        if not isinstance(raw_kernels, Sequence) or isinstance(raw_kernels, (str, bytes)):
            raise ValueError("vae.decoder.stage_kernels must be a sequence")
        stage_kernels = tuple(
            _strict_tuple(
                kernel,
                field=f"vae.decoder.stage_kernels[{index}]",
                length=3,
            )
            for index, kernel in enumerate(raw_kernels)
        )
        if len(stage_kernels) not in {4, 5}:
            raise ValueError("vae.decoder.stage_kernels must contain 4 or 5 kernels")

        raw_upsamples = raw.get("upsamples")
        if not isinstance(raw_upsamples, Sequence) or isinstance(raw_upsamples, (str, bytes)):
            raise ValueError("vae.decoder.upsamples must be a sequence")
        strides: list[tuple[int, int, int]] = []
        reductions: list[int] = []
        for index, upsample in enumerate(raw_upsamples):
            if (
                not isinstance(upsample, Sequence)
                or isinstance(upsample, (str, bytes))
                or len(upsample) != 2
            ):
                raise ValueError(
                    f"vae.decoder.upsamples[{index}] must contain stride and reduction"
                )
            stride = _strict_tuple(
                upsample[0],
                field=f"vae.decoder.upsamples[{index}].stride",
                length=3,
            )
            strides.append((stride[0], stride[1], stride[2]))
            reductions.append(
                _strict_int(
                    upsample[1],
                    field=f"vae.decoder.upsamples[{index}].reduction",
                )
            )
        if len(strides) != 4:
            raise ValueError("vae.decoder.upsamples must contain 4 entries")

        stage5_kernel = _strict_tuple(
            raw.get("stage5_kernel"),
            field="vae.decoder.stage5_kernel",
            length=3,
        )
        if len(stage_kernels) == 5:
            if stage_kernels[-1] != stage5_kernel:
                raise ValueError("vae.decoder stage-5 kernel declarations differ")
            stage_kernels = stage_kernels[:-1]
        patch_size = _strict_int(
            raw.get("patch_size", 4),
            field="vae.decoder.patch_size",
        )
        inferred_fields = ("decoder.t_emb_dim",) if "t_emb_dim" not in raw else ()
        return cls(
            head_dim=_strict_int(raw.get("head_dim"), field="vae.decoder.head_dim"),
            stage_channels=stage_channels,  # type: ignore[arg-type]
            stage_depths=stage_depths,  # type: ignore[arg-type]
            stage_kernels=stage_kernels,  # type: ignore[arg-type]
            upsample_strides=tuple(strides),  # type: ignore[arg-type]
            upsample_channel_reductions=tuple(reductions),  # type: ignore[arg-type]
            stage5_kernel=(stage5_kernel[0], stage5_kernel[1], stage5_kernel[2]),
            patch_size=patch_size,
            t_emb_dim=_strict_int(
                raw.get("t_emb_dim", 384),
                field="vae.decoder.t_emb_dim",
            ),
            timestep_scale_multiplier=_strict_float(
                raw.get("timestep_scale_multiplier", 1000.0),
                field="vae.decoder.timestep_scale_multiplier",
            ),
            model_output_type=str(model_output_type),  # type: ignore[arg-type]
            default_num_inference_steps=_strict_int(
                raw.get("default_num_inference_steps", 1),
                field="vae.decoder.default_num_inference_steps",
            ),
            spatial_compression_ratio=patch_size * math.prod(stride[1] for stride in strides),
            temporal_compression_ratio=math.prod(stride[0] for stride in strides),
            inferred_fields=inferred_fields,
        )


@dataclass(frozen=True)
class VideoVAEConfig:
    """Validated architecture fields required by the native MLX VAE."""

    encoder_blocks: tuple[VideoVAEBlock, ...]
    decoder_blocks: tuple[VideoVAEBlock, ...]
    encoder_base_channels: int = 128
    decoder_base_channels: int = 128
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 128
    patch_size: int = 4
    causal_decoder: bool = False
    timestep_conditioning: bool = False
    spatial_padding_mode: str = "zeros"
    normalize_latent_channels: bool = False
    scaling_factor: float = 1.0
    use_quant_conv: bool = False
    signal_domain: str = "normalized-sdr"
    decoder_kind: Literal["native-conv3d", "diffusion-na"] = "native-conv3d"
    diffusion_decoder: DiffusionVideoDecoderConfig | None = None
    latent_log_var: Literal["uniform", "constant"] = "uniform"
    latent_log_var_value: float | None = None
    inferred_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "encoder_base_channels",
            "decoder_base_channels",
            "in_channels",
            "out_channels",
            "latent_channels",
            "patch_size",
        ):
            _strict_int(getattr(self, field_name), field=field_name)
        _strict_bool(self.causal_decoder, field="causal_decoder")
        _strict_bool(self.timestep_conditioning, field="timestep_conditioning")
        _strict_bool(
            self.normalize_latent_channels,
            field="normalize_latent_channels",
        )
        _strict_float(self.scaling_factor, field="scaling_factor")
        _strict_bool(self.use_quant_conv, field="use_quant_conv")
        if not self.encoder_blocks:
            raise ValueError("encoder_blocks must not be empty")
        if not all(
            isinstance(block, VideoVAEBlock)
            for block in (*self.encoder_blocks, *self.decoder_blocks)
        ):
            raise ValueError("encoder_blocks and decoder_blocks must contain VideoVAEBlock")
        if any(block.name not in _ENCODER_BLOCKS for block in self.encoder_blocks):
            raise ValueError("encoder_blocks contains an unsupported block")
        if any(block.name not in _DECODER_BLOCKS for block in self.decoder_blocks):
            raise ValueError("decoder_blocks contains an unsupported block")
        residual_blocks = [
            block.name
            for block in (*self.encoder_blocks, *self.decoder_blocks)
            if block.residual and block.name != "compress_all"
        ]
        if residual_blocks:
            names = ", ".join(sorted(set(residual_blocks)))
            raise ValueError(
                "residual upsampling is supported only for decoder "
                f"compress_all blocks, got: {names}"
            )
        if self.spatial_padding_mode != "zeros":
            raise ValueError("native video VAE supports only zero spatial padding")
        if self.normalize_latent_channels:
            raise ValueError(
                "native Conv3d video VAE does not implement latent channel renormalization"
            )
        if self.scaling_factor != 1.0:
            raise ValueError(
                f"native Conv3d video VAE requires scaling_factor=1, got {self.scaling_factor}"
            )
        if self.use_quant_conv:
            raise ValueError("native Conv3d video VAE does not implement quantization convolutions")
        if self.signal_domain != "normalized-sdr":
            raise ValueError(
                f"native Conv3d video VAE supports normalized-sdr signal only, got "
                f"{self.signal_domain!r}"
            )
        if self.latent_log_var not in {"uniform", "constant"}:
            raise ValueError("latent_log_var must be 'uniform' or 'constant'")
        if self.latent_log_var == "constant" and self.latent_log_var_value is None:
            raise ValueError("constant latent_log_var requires latent_log_var_value")
        if self.decoder_kind == "native-conv3d":
            if not self.decoder_blocks:
                raise ValueError("native Conv3d decoder_blocks must not be empty")
            if self.diffusion_decoder is not None:
                raise ValueError("native Conv3d config cannot carry diffusion decoder metadata")
            if self.timestep_conditioning:
                raise ValueError("native Conv3d video decoding does not use timestep conditioning")
            if self.latent_log_var != "uniform":
                raise ValueError("native Conv3d video VAE requires uniform latent log variance")
        elif self.decoder_kind == "diffusion-na":
            if self.decoder_blocks:
                raise ValueError("diffusion video VAE cannot carry Conv3d decoder blocks")
            if self.diffusion_decoder is None:
                raise ValueError("diffusion video VAE requires diffusion decoder metadata")
            if not self.timestep_conditioning:
                raise ValueError("diffusion video decoder requires timestep conditioning")
        else:
            raise ValueError(f"unsupported video VAE decoder kind {self.decoder_kind!r}")
        if self.encoder_scale != self.decoder_scale:
            raise ValueError(
                "encoder and decoder scale factors differ: "
                f"{self.encoder_scale} != {self.decoder_scale}"
            )

    @property
    def encoder_scale(self) -> SpatioTemporalScaleFactors:
        return _scale_from_blocks(self.encoder_blocks, patch_size=self.patch_size)

    @property
    def decoder_scale(self) -> SpatioTemporalScaleFactors:
        if self.diffusion_decoder is not None:
            return SpatioTemporalScaleFactors(
                time=self.diffusion_decoder.temporal_compression_ratio,
                height=self.diffusion_decoder.spatial_compression_ratio,
                width=self.diffusion_decoder.spatial_compression_ratio,
            )
        return _scale_from_blocks(self.decoder_blocks, patch_size=self.patch_size)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        """Parse the official checkpoint's ``config.vae`` mapping."""
        if raw.get("_class_name") == "CausalDiffusionVAE":
            return cls._from_diffusion_mapping(raw)
        dims = _strict_int(raw.get("dims", 3), field="vae.dims")
        if dims != 3:
            raise ValueError(f"native video VAE requires dims=3, got {dims}")
        norm_layer = raw.get("norm_layer", "pixel_norm")
        if norm_layer != "pixel_norm":
            raise ValueError(f"native video VAE requires pixel_norm, got {norm_layer!r}")
        latent_log_var = raw.get("latent_log_var", "uniform")
        if latent_log_var != "uniform":
            raise ValueError(
                f"native video VAE requires uniform latent log variance, got {latent_log_var!r}"
            )

        defaulted_fields = (
            "encoder_base_channels",
            "decoder_base_channels",
            "in_channels",
            "out_channels",
            "latent_channels",
            "patch_size",
            "causal_decoder",
            "timestep_conditioning",
            "spatial_padding_mode",
            "normalize_latent_channels",
            "scaling_factor",
            "use_quant_conv",
            "signal_domain",
        )
        return cls(
            encoder_blocks=_parse_blocks(
                raw.get("encoder_blocks"),
                field="vae.encoder_blocks",
                allowed=_ENCODER_BLOCKS,
            ),
            decoder_blocks=_parse_blocks(
                raw.get("decoder_blocks"),
                field="vae.decoder_blocks",
                allowed=_DECODER_BLOCKS,
            ),
            encoder_base_channels=_strict_int(
                raw.get("encoder_base_channels", 128),
                field="vae.encoder_base_channels",
            ),
            decoder_base_channels=_strict_int(
                raw.get("decoder_base_channels", 128),
                field="vae.decoder_base_channels",
            ),
            in_channels=_strict_int(
                raw.get("in_channels", 3),
                field="vae.in_channels",
            ),
            out_channels=_strict_int(
                raw.get("out_channels", 3),
                field="vae.out_channels",
            ),
            latent_channels=_strict_int(
                raw.get("latent_channels", 128),
                field="vae.latent_channels",
            ),
            patch_size=_strict_int(
                raw.get("patch_size", 4),
                field="vae.patch_size",
            ),
            causal_decoder=_strict_bool(
                raw.get("causal_decoder", False),
                field="vae.causal_decoder",
            ),
            timestep_conditioning=_strict_bool(
                raw.get("timestep_conditioning", False),
                field="vae.timestep_conditioning",
            ),
            spatial_padding_mode=str(raw.get("spatial_padding_mode", "zeros")),
            normalize_latent_channels=_strict_bool(
                raw.get("normalize_latent_channels", False),
                field="vae.normalize_latent_channels",
            ),
            scaling_factor=_strict_float(
                raw.get("scaling_factor", 1.0),
                field="vae.scaling_factor",
            ),
            use_quant_conv=_strict_bool(
                raw.get("use_quant_conv", False),
                field="vae.use_quant_conv",
            ),
            signal_domain=str(raw.get("signal_domain", "normalized-sdr")),
            inferred_fields=tuple(f"vae.{field}" for field in defaulted_fields if field not in raw),
        )

    @classmethod
    def _from_diffusion_mapping(cls, raw: Mapping[str, object]) -> Self:
        encoder = raw.get("encoder")
        decoder = raw.get("decoder")
        if not isinstance(encoder, Mapping) or not isinstance(decoder, Mapping):
            raise ValueError("CausalDiffusionVAE requires nested encoder and decoder tables")
        if encoder.get("_class_name") != "Encoder":
            raise ValueError("vae.encoder._class_name must select Encoder")
        dims = _strict_int(encoder.get("dims", 3), field="vae.encoder.dims")
        if dims != 3:
            raise ValueError(f"native video VAE requires dims=3, got {dims}")
        norm_layer = encoder.get("norm_layer", "pixel_norm")
        if norm_layer != "pixel_norm":
            raise ValueError(f"native video VAE requires pixel_norm, got {norm_layer!r}")
        latent_log_var = str(encoder.get("latent_log_var", "constant"))
        if latent_log_var not in {"uniform", "constant"}:
            raise ValueError(f"unsupported latent log variance {latent_log_var!r}")
        latent_log_var_value = encoder.get("latent_log_var_value")
        if latent_log_var_value is not None:
            if isinstance(latent_log_var_value, bool) or not isinstance(
                latent_log_var_value, (int, float)
            ):
                raise ValueError("vae.encoder.latent_log_var_value must be numeric")
            latent_log_var_value = float(latent_log_var_value)

        diffusion = DiffusionVideoDecoderConfig.from_mapping(
            decoder,
            model_output_type=raw.get("model_output_type", "x0"),
        )
        latent_channels = _strict_int(
            encoder.get("out_channels", 128),
            field="vae.encoder.out_channels",
        )
        decoder_latent_channels = _strict_int(
            decoder.get("in_channels", latent_channels),
            field="vae.decoder.in_channels",
        )
        if decoder_latent_channels != latent_channels:
            raise ValueError("video VAE encoder and decoder latent channels differ")
        patch_size = _strict_int(
            encoder.get("patch_size", decoder.get("patch_size", 4)),
            field="vae.patch_size",
        )
        if (
            _strict_int(
                decoder.get("patch_size", patch_size),
                field="vae.decoder.patch_size",
            )
            != patch_size
        ):
            raise ValueError("video VAE encoder and decoder patch sizes differ")

        return cls(
            encoder_blocks=_parse_blocks(
                encoder.get("blocks"),
                field="vae.encoder.blocks",
                allowed=_ENCODER_BLOCKS,
            ),
            decoder_blocks=(),
            encoder_base_channels=_strict_int(
                encoder.get("base_channels", 128),
                field="vae.encoder.base_channels",
            ),
            decoder_base_channels=diffusion.stage_channels[-1],
            in_channels=_strict_int(
                encoder.get("in_channels", 3),
                field="vae.encoder.in_channels",
            ),
            out_channels=_strict_int(
                decoder.get("out_channels", 3),
                field="vae.decoder.out_channels",
            ),
            latent_channels=latent_channels,
            patch_size=patch_size,
            causal_decoder=False,
            timestep_conditioning=True,
            spatial_padding_mode=str(encoder.get("spatial_padding_mode", "zeros")),
            normalize_latent_channels=False,
            scaling_factor=1.0,
            use_quant_conv=False,
            signal_domain=str(raw.get("signal_domain", "normalized-sdr")),
            decoder_kind="diffusion-na",
            diffusion_decoder=diffusion,
            latent_log_var=latent_log_var,  # type: ignore[arg-type]
            latent_log_var_value=latent_log_var_value,
            inferred_fields=(("vae.signal_domain",) if "signal_domain" not in raw else ()),
        )

    @classmethod
    def from_checkpoint(cls, path: Path | str) -> Self:
        """Read and validate VAE architecture metadata from safetensors."""
        metadata = read_metadata(path)
        try:
            raw_config = json.loads(metadata["config"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}: missing or invalid config.vae metadata") from exc
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"{path}: config must be a table")
        raw_vae = raw_config.get("vae", raw_config)
        if not isinstance(raw_vae, Mapping):
            raise ValueError(f"{path}: config.vae must be a table")
        return cls.from_mapping(raw_vae)


def _res(num_layers: int) -> VideoVAEBlock:
    return VideoVAEBlock(name="res_x", num_layers=num_layers)


def _compress(name: str, multiplier: int) -> VideoVAEBlock:
    return VideoVAEBlock(name=name, multiplier=multiplier)


# LTX-2.3 architecture facts as carried by the checkpoint's config.vae
# metadata; every scalar field matches the dataclass defaults.
LTX23_VIDEO_VAE_CONFIG = VideoVAEConfig(
    encoder_blocks=(
        _res(4),
        _compress("compress_space_res", 2),
        _res(6),
        _compress("compress_time_res", 2),
        _res(4),
        _compress("compress_all_res", 2),
        _res(2),
        _compress("compress_all_res", 1),
        _res(2),
    ),
    decoder_blocks=(
        _res(4),
        _compress("compress_space", 2),
        _res(6),
        _compress("compress_time", 2),
        _res(4),
        _compress("compress_all", 1),
        _res(2),
        _compress("compress_all", 2),
        _res(2),
    ),
)


__all__ = [
    "DiffusionVideoDecoderConfig",
    "LTX23_VIDEO_VAE_CONFIG",
    "VideoVAEBlock",
    "VideoVAEConfig",
]
