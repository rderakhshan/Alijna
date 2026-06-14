#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. PersonalAI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import argparse
import json
import logging
import time
from tqdm import tqdm
import threading
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from FlashOAgents import OpenAIServerModel
from base_agent import SearchAgent
from utils import read_jsonl, write_jsonl

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

def process_item(item, model, summary_interval, prompts_type, max_steps, memory_provider=None, direct_output_dir=None, item_id=None):

    search_agent = SearchAgent(
        model, 
        summary_interval=summary_interval, 
        prompts_type=prompts_type, 
        max_steps=max_steps
    )

    question = item["question"]
    golden_answer = item["answer"]

    model.reset_cumulative_tokens()
    start_time = time.monotonic()
    try:
        result = search_agent(question)
    except Exception as e:
        logger.error(f"Exception occurred while calling multi_agent: {str(e)}")
        return None
    elapsed = time.monotonic() - start_time
    token_counts = model.get_cumulative_token_counts()

    output = {
        "question": question,
        "golden_answer": golden_answer,
        **result,
        "metrics": {
            "elapsed_time": elapsed,
            "total_tokens": token_counts["total_tokens"],
            "prompt_tokens": token_counts["input_token_count"],
            "completion_tokens": token_counts["output_token_count"],
        },
    }

    if memory_provider:
        output["memory_provider"] = memory_provider

    if direct_output_dir and item_id is not None:
        task_file = Path(direct_output_dir) / f"{item_id}.json"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        with open(task_file, "w") as f:
            json.dump(output, f, indent=2)

    return output


def main(args):
    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
    model = OpenAIServerModel(
        os.environ.get("DEFAULT_MODEL"),
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=32768,
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("OPENAI_API_BASE"),
    )

    if args.infile.lower().endswith('.json'):
        with open(args.infile, 'r') as f:
            data = json.load(f)
    else:
        data = read_jsonl(args.infile)

    if args.sample_num is not None:
        data = data[:args.sample_num]

    task_indices = None
    if args.task_indices:
        task_indices = [int(i) - 1 for i in args.task_indices.split(",")]

    if task_indices is not None:
        data = [data[i] for i in task_indices if i < len(data)]
    
    try:
        out_data = read_jsonl(args.outfile)
    except Exception:
        out_data = []
    done_questions = set([item.get("question") for item in out_data])
    data_to_run = [item for item in data if item.get("question") not in done_questions]
    logger.info(f"Total data: {len(data)}, Completed: {len(done_questions)}, Remaining: {len(data_to_run)}")

    results = []
    file_lock = threading.Lock()

    def safe_write(result):
        with file_lock:
            write_jsonl(args.outfile, [result], "a")

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        summary_interval = random.randint(args.summary_interval - 1, args.summary_interval + 1)

        futures = [
            executor.submit(
                process_item, 
                item, 
                model, 
                summary_interval, 
                args.prompts_type, 
                args.max_steps,
                memory_provider=args.memory_provider,
                direct_output_dir=args.direct_output_dir,
                item_id=str(i) if args.direct_output_dir else None,
            ) for i, item in enumerate(data_to_run)
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result:
                results.append(result)
                safe_write(result)

    logger.info(f"Processing completed. Newly added: {len(results)}, Total completed: {len(done_questions) + len(results)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data generation script')

    parser.add_argument('--infile', type=str, default="./data/<example.json>", help='input path')
    parser.add_argument('--outfile', type=str, default="./output/<example.jsonl>", help='output path')
    parser.add_argument('--sample_num', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--summary_interval', type=int, default=8, help='Summary interval')
    parser.add_argument('--prompts_type', type=str, default="default", help='Type of prompts to use')
    parser.add_argument('--concurrency', type=int, default=15, help='Number of concurrency')
    parser.add_argument('--max_steps', type=int, default=40, help='Maximum number of steps')
    parser.add_argument('--task_indices', type=str, default=None, help='Comma-separated 1-based task indices to process')
    parser.add_argument('--memory_provider', type=str, default=None, help='Memory provider name for evolution')
    parser.add_argument('--direct_output_dir', type=str, default=None, help='Directory for per-task JSON output files')

    args = parser.parse_args()
    
    main(args)
    