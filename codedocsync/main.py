"""
Main entry point for the CodeDocSync CLI application.

This is a compatibility wrapper that imports from the refactored CLI module.
All functionality has been moved to the cli subpackage for better organization.
"""

from codedocsync.cli.main import app

if __name__ == "__main__":
    app()
