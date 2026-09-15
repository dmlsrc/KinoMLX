"""Krea2 canonical adapter collection declaration."""

from kinomlx.adapters.config_schema import adapter_collection_field, adapter_record_spec

from .inputs import Krea2AdapterConfig

ADAPTER_RECORD = adapter_record_spec(Krea2AdapterConfig)
ADAPTERS = adapter_collection_field(ADAPTER_RECORD)
