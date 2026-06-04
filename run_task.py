#!/usr/bin/env python3
"""CLI entry point: python run_task.py --input provided_inputs/task_01_firelight.json"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline import run_pipeline_from_json


def main():
    parser = argparse.ArgumentParser(description="Boock Theme Song Generator")
    parser.add_argument("--input", required=True, help="Path to task JSON file")
    parser.add_argument("--outputs-root", default=None, help="Root output directory (default: ./outputs)")
    parser.add_argument("--seed", type=int, default=42, help="YuE sampling seed (re-roll to escape a bad generation trajectory)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Running theme song generation for: {args.input}")
    response = run_pipeline_from_json(args.input, outputs_root=args.outputs_root, seed=args.seed)

    print(json.dumps(response.model_dump(), indent=2))

    if response.warnings:
        print("\nWarnings:", file=sys.stderr)
        for w in response.warnings:
            print(f"  - {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
