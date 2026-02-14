"""Allow running RLM as a module: python -m rlm."""

import sys

from .cli import main

sys.exit(main())
