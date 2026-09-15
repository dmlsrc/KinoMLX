"""Early admission for the current one/two-reference editing recipe."""

from kinomlx.config_rules import validate_record

from .request_rules import admit_recipe_mode
from .types import Krea2Request


def validate_edit_request(request: Krea2Request) -> None:
    validate_record(request, location="[generate]")
    admit_recipe_mode(request, "edit")
    if request.reference_boost_mask is not None and not request.reference_boost_mask.is_file():
        raise ValueError(f"reference boost mask is not a file: {request.reference_boost_mask}")
