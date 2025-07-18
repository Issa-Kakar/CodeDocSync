"""Configuration management for CodeDocSync."""

from pydantic import BaseModel, Field, validator
from typing import List, Dict
import re


class MatcherConfig(BaseModel):
    """Configuration for the matching system."""

    # Direct matcher settings
    enable_fuzzy: bool = Field(default=True, description="Enable fuzzy name matching")
    fuzzy_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Minimum similarity for fuzzy matches"
    )

    # Performance settings
    max_line_distance: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum line distance for fuzzy match validation",
    )

    # Pattern matching
    custom_patterns: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Custom regex patterns for name transformation",
    )

    @validator("custom_patterns")
    def validate_patterns(cls, patterns):
        """Validate regex patterns are valid."""
        for pattern_dict in patterns:
            if "pattern" not in pattern_dict or "replacement" not in pattern_dict:
                raise ValueError("Pattern dict must have 'pattern' and 'replacement'")
            try:
                re.compile(pattern_dict["pattern"])
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")
        return patterns


class ParserConfig(BaseModel):
    """Configuration for the parser system."""

    docstring_style: str = Field(default="auto", description="Default docstring style")
    include_private: bool = Field(
        default=False, description="Include private functions"
    )


class AnalysisConfig(BaseModel):
    """Configuration for the analysis system."""

    llm_provider: str = Field(default="openai", description="LLM provider")
    model: str = Field(default="gpt-4o-mini", description="Model name")


class CodeDocSyncConfig(BaseModel):
    """Complete configuration for CodeDocSync."""

    version: int = 1
    parser: ParserConfig = Field(default_factory=ParserConfig)
    matcher: MatcherConfig = Field(default_factory=MatcherConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

    @classmethod
    def from_yaml(cls, file_path: str) -> "CodeDocSyncConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
