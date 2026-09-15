"""Linear adapters with explicitly saved shared projection bases."""

import math

import mlx.core as mx

from .formats import AdapterOperation


def _shape(value: mx.array, shape: tuple[int, ...], name: str) -> mx.array:
    if value.shape != shape:
        raise ValueError(f"{name} shape mismatch: got {value.shape}, expected {shape}")
    return value.astype(mx.float32)


def basis_delta(operation: AdapterOperation, base: mx.array) -> mx.array:
    options = operation.linear_options
    if options is None or base.ndim != 2 or operation.part != "weight":
        raise ValueError(f"{operation.kind} requires a configured Linear weight target")
    out, inp = base.shape
    rank = options.rank
    factors = operation.factors
    if operation.kind == "tinylora":
        down = _shape(factors["A"], (rank, inp), "TinyLoRA A")
        up = _shape(factors["B"], (out, rank), "TinyLoRA B")
        rotation = mx.einsum(
            "i,ijk->jk", factors["v"].astype(mx.float32), factors["P"].astype(mx.float32)
        )
        return (up @ rotation) @ down
    if operation.kind in ("vera", "pvera"):
        count = rank * (2 if operation.kind == "pvera" else 1)
        down, up = factors["A"], factors["B"]
        if (
            down.ndim != 2
            or up.ndim != 2
            or down.shape[0] != count
            or up.shape[1] != rank
            or down.shape[1] < inp
            or up.shape[0] < out
        ):
            raise ValueError(f"{operation.kind} shared projections cannot cover this target")
        scale_d = _shape(factors["lambda_d"], (count,), f"{operation.kind} lambda_d")[:rank]
        scale_b = _shape(factors["lambda_b"], (out,), f"{operation.kind} lambda_b")
        return (up[:out].astype(mx.float32) * scale_b[:, None]) @ (
            down[:rank, :inp].astype(mx.float32) * scale_d[:, None]
        )
    if operation.kind == "randlora":
        small, large = min(inp, out), max(inp, out)
        count = math.ceil(small / rank)
        down, up = factors["A"], factors["B"]
        if (
            down.ndim != 3
            or up.ndim != 3
            or down.shape[0] != rank
            or up.shape[2] != rank
            or down.shape[2] < small
            or up.shape[0] < large
            or up.shape[1] < count
            or (down.shape[1] != 1 and down.shape[1] < count)
        ):
            raise ValueError("RandLoRA shared projections cannot cover this target")
        gain = _shape(factors["lambda"], (rank, count), "RandLoRA lambda")
        gamma = _shape(factors["gamma"], (count, small), "RandLoRA gamma")
        a = down[:, :count, :small].astype(mx.float32) * gain[..., None] * gamma[None, ...]
        b = up[:large, :count].astype(mx.float32)
        # The serialized equation flattens A as (rank, basis) and B as
        # (basis, rank). Preserve that ordering, including unequal dimensions.
        delta = b.reshape(large, -1) @ a.reshape(-1, small)
        return (delta if inp == small else delta.T) * options.scale
    if operation.kind == "unilora":
        theta = factors["theta_d"].astype(mx.float32)
        matrices = []
        for side, shape in (("A", (rank, inp)), ("B", (out, rank))):
            indices = factors[f"indices_{side}"]
            if indices.shape != shape:
                raise ValueError(f"UniLoRA {side} indices do not match the target")
            scales = _shape(factors[f"scales_{side}"], shape, f"UniLoRA {side} scales")
            matrices.append(theta[indices.astype(mx.int32)] * scales)
        return matrices[1] @ matrices[0]
    if operation.kind != "vblora":
        raise ValueError(f"{operation.kind} has no shared-basis equation")
    width = options.block_size
    if inp % width or out % width:
        raise ValueError("VBLoRA vector_length must divide input and output widths")
    bank = factors["vector_bank"].astype(mx.float32)
    matrices = []
    for side, shape in (("A", (rank, inp // width)), ("B", (out // width, rank))):
        if options.variant == "compressed":
            indices = factors[f"logits_{side}_topk_indices"]
            if indices.shape != (*shape, options.topk):
                raise ValueError(f"VBLoRA {side} indices do not match the target")
            indices = indices.astype(mx.int32)
            weights = factors[f"logits_{side}_topk_weights"].astype(mx.float32)
            probability = mx.concatenate(
                [weights, 1.0 - mx.sum(weights, axis=-1, keepdims=True)], axis=-1
            )
        else:
            logits = factors[f"logits_{side}"].astype(mx.float32)
            if logits.shape != (*shape, bank.shape[0]):
                raise ValueError(f"VBLoRA {side} logits do not match the target")
            indices = mx.argsort(logits, axis=-1)[..., -options.topk :][..., ::-1]
            probability = mx.softmax(mx.take_along_axis(logits, indices, axis=-1), axis=-1)
        mixture = mx.sum(bank[indices] * probability[..., None], axis=-2)
        matrices.append(
            mixture.reshape(rank, inp)
            if side == "A"
            else mixture.transpose(0, 2, 1).reshape(out, rank)
        )
    return matrices[1] @ matrices[0]
