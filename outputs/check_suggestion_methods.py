import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import inspect

from codedocsync.suggestions.generators.parameter_generator import (
    ParameterSuggestionGenerator,
)

# Check what methods actually exist
gen = ParameterSuggestionGenerator()
methods = [m for m in dir(gen) if not m.startswith("_")]
print("Available methods on ParameterSuggestionGenerator:")
for method in methods:
    print(f"  - {method}")

# Check if it has generate or generate_suggestion
if hasattr(gen, "generate"):
    print("\n[PASS] Has 'generate' method")
    print(f"  Signature: {inspect.signature(gen.generate)}")

if hasattr(gen, "generate_suggestion"):
    print("\n[PASS] Has 'generate_suggestion' method")
    print(f"  Signature: {inspect.signature(gen.generate_suggestion)}")
