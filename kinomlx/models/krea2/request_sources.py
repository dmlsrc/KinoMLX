"""Explicit saved-conditioning observations for a surviving Krea request."""

from kinomlx.config_records import derive_config
from kinomlx.config_rules import validate_record
from kinomlx.prompts import is_empty_prompt_batch, normalize_prompts

from .text_conditioning import read_text_conditioning_prompts
from .types import Krea2Request


def resolve_request_sources(request: Krea2Request, *, location: str = "[generate]") -> Krea2Request:
    """Read surviving source context without turning inherited rows into supplied inputs.

    Input fields remain the canonical request. The returned record's non-init rows
    are observations for execution; every source-consuming boundary calls this step.
    Construction and ordinary typed copies perform no file observations.
    ``location`` names the table a violation should be reported against, which
    is the replayed source table when a parent record is being resolved.
    """
    validate_record(request, location=location)
    request.resolved_adapters()
    if request.text_conditioning is None:
        if not request.prompts:
            raise ValueError("prompt is required unless text_conditioning supplies it")
        return request
    saved_prompts, saved_negative = read_text_conditioning_prompts(request.text_conditioning)
    unconditional = request.prompt_unconditional
    if request.prompt is None and is_empty_prompt_batch(saved_prompts):
        unconditional = True
    if unconditional and not is_empty_prompt_batch(saved_prompts):
        raise ValueError("prompt_unconditional requires empty saved text conditioning")
    resolved = derive_config(request, prompt_unconditional=unconditional)
    prompts = normalize_prompts(
        saved_prompts if request.prompt is None else request.prompt,
        name="prompt",
        allow_empty=unconditional,
    )
    negative_value = saved_negative if request.negative_prompt is None else request.negative_prompt
    negative = (
        None
        if negative_value is None
        else normalize_prompts(
            negative_value,
            name="negative_prompt",
            allow_empty=True,
        )
    )
    object.__setattr__(resolved, "_resolved_prompts", prompts)
    object.__setattr__(resolved, "_resolved_negative_prompts", negative)
    return resolved
