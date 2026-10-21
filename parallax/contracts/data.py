"""Dataset contract and normalized schema."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DatasetSample(BaseModel):
    """Normalized schema for prompt-injection and jailbreak dataset samples.

    Fields specified in DESIGN.md Section 6:
    text, label_multiclass, label_binary, source, split.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="Prompt text payload")
    label_multiclass: str = Field(
        ...,
        description="Technique label or category (e.g., 'direct_instruction', 'none')",
    )
    label_binary: bool = Field(
        ...,
        description="Binary label: True if attack/jailbreak/injection, False if benign",
    )
    source: str = Field(..., description="Dataset or generator source identifier")
    split: str = Field(
        ..., description="Dataset split: 'train', 'val', or 'test'"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional auxiliary attributes (e.g., original class, generator seed)",
    )
