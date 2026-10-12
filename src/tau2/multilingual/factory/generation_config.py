"""The one generation-settings shape shared by every factory authoring call."""

from typing import Annotated, Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

FactoryReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]


class FactoryModelConfig(BaseModel):
    """Explicit authoring settings, separate from the benchmark customer model.

    Reasoning effort is always pinned; there is no unpinned or "none" setting.
    """

    model_config = ConfigDict(extra="forbid")

    model: Annotated[str, Field(min_length=1, description="Authoring generation model")]
    reasoning_effort: Annotated[
        FactoryReasoningEffort,
        Field(description="Explicit reasoning setting for this authoring request"),
    ]

    def provenance_args(self) -> dict[str, str]:
        """Model arguments exactly as recorded in generation provenance."""
        return {"reasoning_effort": self.reasoning_effort}


def regional_generation(
    model: str, model_args: Mapping[str, Any]
) -> FactoryModelConfig:
    """Map a regional-review request's recorded settings onto the factory shape.

    Banking/intake review requests record provider arguments verbatim. Regional
    map drafting pins only the model and reasoning effort; ``temperature`` is
    not a factory setting and is not forwarded. Any other argument is refused.
    """
    unknown = set(model_args) - {"reasoning_effort", "temperature"}
    if unknown:
        raise ValueError(f"Unsupported regional drafting arguments: {sorted(unknown)}")
    if "reasoning_effort" not in model_args:
        raise ValueError("Regional drafting requires an explicit reasoning_effort")
    return FactoryModelConfig(
        model=model, reasoning_effort=model_args["reasoning_effort"]
    )
