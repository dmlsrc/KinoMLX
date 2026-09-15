"""Final adapter relationships with explicit supplied-strength ownership."""

from kinomlx.config_rules import RecordRule, RuleInputs


def _schedule(values: RuleInputs) -> None:
    if values["step_strengths"] and values.presence["strength"]:
        raise ValueError("strength and step_strengths cannot both be supplied")
    if values["step_strengths"] and values["sigma_window"]:
        raise ValueError("sigma_window and step_strengths cannot both be supplied")


ADAPTER_SCHEDULE = RecordRule(
    "adapter_schedule",
    frozenset({"strength", "step_strengths", "sigma_window"}),
    _schedule,
    presence_reads=frozenset({"strength"}),
)
