#!/bin/bash
set -e
mkdir -p output

# set environment variables
export SERPER_API_KEY=""    # web search
export JINA_API_KEY=""      # crawl pages
export OPENAI_API_KEY=""
export OPENAI_API_BASE=""
export DEFAULT_MODEL=""

uv run python run_flash_searcher.py \
    --infile data/example.jsonl \
    --outfile output/example.jsonl \
    --summary_interval 8 \
    --concurrency 15 \
    --max_steps 40