"""Matcher module for finding function-documentation pairs."""

from .models import MatchedPair, MatchConfidence, MatchType, MatchResult, MatchingError
from .direct_matcher import DirectMatcher
from .facade import MatchingFacade
from .contextual_models import (
    ImportType,
    ImportStatement,
    ModuleInfo,
    FunctionLocation,
    CrossFileMatch,
    ContextualMatcherState,
)
from .contextual_matcher import ContextualMatcher
from .contextual_facade import ContextualMatchingFacade
from .import_parser import ImportParser
from .module_resolver import ModuleResolver
from .function_registry import FunctionRegistry
from .doc_location_finder import DocLocationFinder
from .semantic_models import (
    EmbeddingModel,
    EmbeddingConfig,
    FunctionEmbedding,
    SemanticMatch,
    SemanticSearchResult,
)

__all__ = [
    "MatchedPair",
    "MatchConfidence",
    "MatchType",
    "MatchResult",
    "DirectMatcher",
    "MatchingError",
    "MatchingFacade",
    "ImportType",
    "ImportStatement",
    "ModuleInfo",
    "FunctionLocation",
    "CrossFileMatch",
    "ContextualMatcherState",
    "ContextualMatcher",
    "ContextualMatchingFacade",
    "ImportParser",
    "ModuleResolver",
    "FunctionRegistry",
    "DocLocationFinder",
    "EmbeddingModel",
    "EmbeddingConfig",
    "FunctionEmbedding",
    "SemanticMatch",
    "SemanticSearchResult",
]
