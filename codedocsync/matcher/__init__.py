"""Matcher module for finding function-documentation pairs."""

from .contextual_facade import ContextualMatchingFacade
from .contextual_matcher import ContextualMatcher
from .contextual_models import (
    ContextualMatcherState,
    CrossFileMatch,
    FunctionLocation,
    ImportStatement,
    ImportType,
    ModuleInfo,
)
from .direct_matcher import DirectMatcher
from .doc_location_finder import DocLocationFinder
from .facade import MatchingFacade
from .function_registry import FunctionRegistry
from .import_parser import ImportParser
from .models import MatchConfidence, MatchedPair, MatchingError, MatchResult, MatchType
from .module_resolver import ModuleResolver
from .semantic_matcher import SemanticMatcher
from .semantic_models import (
    EmbeddingConfig,
    EmbeddingModel,
    FunctionEmbedding,
    SemanticMatch,
    SemanticSearchResult,
)
from .unified_facade import UnifiedMatchingFacade

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
    "SemanticMatcher",
    "UnifiedMatchingFacade",
]
