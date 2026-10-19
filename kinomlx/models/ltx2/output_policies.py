"""LTX output availability derived from the stage model, by artifact and by field."""

from kinomlx.config.stages import OutputPolicy, produced_at_stages

from .restart_rules import (
    FINAL_LATENTS,
    PRODUCES,
    READABLE,
    STAGE_1_CONDITIONING,
    STAGE_1_LATENTS,
    STAGE_2_CONDITIONING,
    STAGES,
    TEXT_CONDITIONING,
)

ARTIFACT_STAGES = frozendict(
    {
        name: OutputPolicy(stages)
        for name, stages in produced_at_stages(STAGES, PRODUCES, READABLE).items()
    }
)
ARTIFACT_GROUPS = frozendict(
    {
        "save_latents": (STAGE_1_LATENTS, FINAL_LATENTS),
        "save_text_conditioning": (TEXT_CONDITIONING,),
        "save_media_conditioning": (STAGE_1_CONDITIONING, STAGE_2_CONDITIONING),
    }
)
OUTPUT_POLICIES = frozendict(
    {
        field: OutputPolicy(frozenset(*ARTIFACT_STAGES[name].produced_at for name in artifacts))
        for field, artifacts in ARTIFACT_GROUPS.items()
    }
)
