#!/usr/bin/env python3
"""Entrypoint wrapper for running the Deep Agent from the command-line."""

import os
import sys

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(override=True)

# Add src to the path so the deep_agents_from_scratch package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from deep_agents_from_scratch.cli import main

if __name__ == "__main__":
    main()
