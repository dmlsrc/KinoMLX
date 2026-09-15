"""Exercise the runtime fuser and its separate post-use validation in algebra tests."""

import mlx.core as mx

from kinomlx.adapters.fusion import fuse_adapter_operations
from kinomlx.adapters.validation import AdapterWeightCheck
from kinomlx.reporting import NullReporter


def fuse_and_check(targets, operations, *, transposed=frozenset(), accumulation="float32"):
    pending = AdapterWeightCheck()
    fuse_adapter_operations(
        targets, operations, pending=pending, transposed=transposed, accumulation=accumulation
    )
    mx.eval([getattr(target.owner, target.attribute) for target in targets.values()])
    pending.run(context="adapter algebra", reporter=NullReporter())
