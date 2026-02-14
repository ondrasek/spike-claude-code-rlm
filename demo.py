#!/usr/bin/env python3
"""Demo script for RLM (Recursive Language Model).

This is a convenience wrapper around the package CLI.
For direct usage, prefer: uvx rlm [options]
Or: python -m rlm [options]
"""

import sys

from rlm.cli import main

if __name__ == "__main__":
    sys.exit(main())
