"""Local Klein bounds shared by request fields and resolved recipe helpers."""

from enum import StrEnum

from kinomlx.config_constraints import ChoiceConstraint, IntegerConstraint, NumberConstraint

OUTPUT_DIMENSION = IntegerConstraint(minimum=64)
DENOISE_STEPS = IntegerConstraint(minimum=1)
GUIDANCE = NumberConstraint(minimum=1)


class Flux2CFGMode(StrEnum):
    """How the two classifier-free guidance branches are evaluated."""

    SEQUENTIAL = "sequential"
    BATCHED = "batched"


CFG_MODE_CHOICES = tuple(item.value for item in Flux2CFGMode)
CFG_MODE = ChoiceConstraint(CFG_MODE_CHOICES)
