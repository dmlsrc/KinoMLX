"""LTX-2 canonical adapter collection and stage-specific shorthand declaration."""

from kinomlx.adapters.config_schema import (
    adapter_collection_field,
    adapter_record_spec,
    adapter_strength_field,
)

from .inputs import LTX2AdapterConfig

_STAGE1 = adapter_strength_field("stage1_strength", "Override this adapter's stage-1 strength")
_STAGE2 = adapter_strength_field("stage2_strength", "Override this adapter's stage-2 strength")
ADAPTER_RECORD = adapter_record_spec(LTX2AdapterConfig, extra_fields=(_STAGE1, _STAGE2))
ADAPTERS = adapter_collection_field(ADAPTER_RECORD, stage_fields=(_STAGE1, _STAGE2))
