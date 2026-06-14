#!/usr/bin/env python
# coding=utf-8

import sys
from pathlib import Path

# Add Flash_Searcher to search path
sys.path.insert(0, str(Path(__file__).resolve().parent / "Flash_Searcher"))

from run_flash_searcher_mm import main, parser

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
