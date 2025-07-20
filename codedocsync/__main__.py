"""
Allow CodeDocSync to be invoked as a module.

This enables running the CLI with:
    python -m codedocsync
"""

from codedocsync.cli.main import app

if __name__ == "__main__":
    app()
