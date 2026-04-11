"""Vercel serverless entry point for AI Scout Web UI."""

import sys
from pathlib import Path

# Add project root to path so aiscout package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from aiscout.web.app import app
