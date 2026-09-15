"""Local Klein bounds shared by request fields and resolved recipe helpers."""

from kinomlx.config_constraints import ChoiceConstraint, IntegerConstraint, NumberConstraint

OUTPUT_DIMENSION = IntegerConstraint(minimum=64)
DENOISE_STEPS = IntegerConstraint(minimum=1)
GUIDANCE = NumberConstraint(minimum=1)
CFG_MODE = ChoiceConstraint(("sequential", "batched"))
