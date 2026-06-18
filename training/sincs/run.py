# Copyright 2024 Cerebras Systems.
#
# SINCS model runner for Cerebras CSX

import os
import sys

# Add modelzoo src to path (5 levels up: sincs -> vision -> models -> modelzoo -> cerebras -> src)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from cerebras.modelzoo.common.run_utils import run


def main():
    run()


if __name__ == "__main__":
    main()
