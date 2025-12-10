#!/usr/bin/env python3
"""
Copy portions of real PyTorch profiler traces for testing.
This approach uses actual trace data to ensure TraceLens compatibility.
"""

import json
import shutil
from pathlib import Path
import argparse
from typing import Dict, List
import random


class RealTraceCopier:
    """Copy portions of real traces to create test data."""

    def __init__(self, source_dir: str, output_dir: str = "testdata"):
        """
        Initialize the trace copier.

        Args:
            source_dir: Path to real experiment data with traces
            output_dir: Output directory for test data
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.sweep_dir = self.output_dir / "test_sweep"

        # Test configurations
        self.thread_configs = [256, 512]
        self.channel_configs = [28, 56]
        self.num_ranks = 7  # Use ranks 0-6
        self.num_batches_to_copy = 5  # Copy first 5 ProfilerSteps

    def extract_batches_from_trace(self, trace_file: Path, num_batches: int) -> Dict:
        """
        Extract the first N batches (ProfilerSteps) from a trace file.

        Args:
            trace_file: Path to the trace JSON file
            num_batches: Number of ProfilerSteps to extract

        Returns:
            Modified trace data with only the requested batches
        """
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)

        events = trace_data.get('traceEvents', [])

        # Find ProfilerStep events
        profiler_steps = []
        for event in events:
            if event.get('cat') == 'user_annotation' and 'ProfilerStep#' in event.get('name', ''):
                profiler_steps.append(event)

        # Sort by timestamp
        profiler_steps.sort(key=lambda x: x['ts'])

        # Take first N steps
        selected_steps = profiler_steps[:num_batches]

        if not selected_steps:
            print(f"  Warning: No ProfilerSteps found in {trace_file}")
            return trace_data

        # Get time range for selected steps
        min_ts = min(step['ts'] for step in selected_steps)
        max_ts = max(step['ts'] + step.get('dur', 0) for step in selected_steps)

        # Filter events within this time range
        filtered_events = []
        for event in events:
            event_ts = event.get('ts', 0)
            event_end = event_ts + event.get('dur', 0)

            # Include event if it overlaps with our time range
            if event_ts <= max_ts and event_end >= min_ts:
                filtered_events.append(event)

        # Update trace data
        trace_data['traceEvents'] = filtered_events

        # Add metadata about extraction
        if 'metadata' not in trace_data:
            trace_data['metadata'] = {}
        trace_data['metadata']['extracted_batches'] = num_batches
        trace_data['metadata']['original_file'] = str(trace_file)

        print(f"    Extracted {len(filtered_events)} events covering {num_batches} batches")

        return trace_data

    def find_source_traces(self) -> Dict[str, Dict[str, Path]]:
        """
        Find available source trace files in the experiment directory.

        Returns:
            Dictionary mapping config -> rank -> trace path
        """
        traces = {}

        # Look for traces in the source directory
        for thread_dir in self.source_dir.glob("*thread"):
            thread_num = int(thread_dir.name.replace('thread', ''))

            for channel_dir in thread_dir.glob("nccl_*channels"):
                channel_num = int(channel_dir.name.replace('nccl_', '').replace('channels', ''))

                config_key = f"{thread_num}_{channel_num}"
                traces[config_key] = {}

                # Find torch_profiler traces
                profiler_dir = channel_dir / "torch_profiler"
                if profiler_dir.exists():
                    for rank_dir in profiler_dir.glob("rank*"):
                        rank_num = int(rank_dir.name.replace('rank', ''))

                        # Look for customer trace files
                        trace_files = list(rank_dir.glob("customer_trace_step*.json"))
                        if trace_files:
                            # Use the first available trace
                            traces[config_key][rank_num] = trace_files[0]

        return traces

    def copy_traces(self) -> None:
        """Copy and process traces from source to test data."""
        print(f"Copying real traces from: {self.source_dir}")
        print(f"Output directory: {self.sweep_dir}")
        print(f"Extracting first {self.num_batches_to_copy} batches per trace\n")

        # Find available source traces
        source_traces = self.find_source_traces()

        if not source_traces:
            print("ERROR: No source traces found in", self.source_dir)
            print("Please provide a valid experiment directory with PyTorch traces")
            return

        print(f"Found traces for configs: {list(source_traces.keys())}\n")

        # Create test data for each configuration
        for thread in self.thread_configs:
            for channel in self.channel_configs:
                config_name = f"{thread}thread/nccl_{channel}channels"
                config_key = f"{thread}_{channel}"

                print(f"Processing {config_name}...")

                # Check if we have source data for this config
                if config_key not in source_traces:
                    # Use data from a similar config
                    alt_keys = [k for k in source_traces.keys() if str(channel) in k]
                    if alt_keys:
                        config_key = alt_keys[0]
                        print(f"  Using traces from config {config_key} as substitute")
                    else:
                        # Use any available config
                        config_key = list(source_traces.keys())[0]
                        print(f"  Using traces from config {config_key} as substitute")

                source_config = source_traces[config_key]

                # Copy traces for each rank
                for rank in range(self.num_ranks):
                    output_dir = self.sweep_dir / config_name / f"torch_profiler/rank{rank}"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Find source trace for this rank or use another rank
                    if rank in source_config:
                        source_trace = source_config[rank]
                    else:
                        # Use trace from another rank
                        available_ranks = list(source_config.keys())
                        if available_ranks:
                            alt_rank = available_ranks[rank % len(available_ranks)]
                            source_trace = source_config[alt_rank]
                            print(f"  Rank {rank}: Using rank {alt_rank} data")
                        else:
                            print(f"  Rank {rank}: No source data available")
                            continue

                    # Extract batches from the trace
                    trace_data = self.extract_batches_from_trace(
                        source_trace,
                        self.num_batches_to_copy
                    )

                    # Save to output
                    output_file = output_dir / "customer_trace_step10.json"
                    with open(output_file, 'w') as f:
                        json.dump(trace_data, f, indent=2)

                    print(f"    Rank {rank}: Saved to {output_file.relative_to(self.output_dir)}")

                print()

        # Create metadata file
        metadata = {
            "source_directory": str(self.source_dir),
            "thread_configs": self.thread_configs,
            "channel_configs": self.channel_configs,
            "num_ranks": self.num_ranks,
            "batches_per_trace": self.num_batches_to_copy,
            "description": "Test data created by copying portions of real PyTorch profiler traces"
        }

        with open(self.sweep_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nTest data created successfully!")
        print(f"  Thread configs: {self.thread_configs}")
        print(f"  Channel configs: {self.channel_configs}")
        print(f"  Number of ranks: {self.num_ranks}")
        print(f"  Batches per trace: {self.num_batches_to_copy}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy portions of real PyTorch traces for testing"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Source directory containing real experiment traces"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="testdata",
        help="Output directory for test data (default: testdata)"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="Number of ProfilerSteps to copy per trace (default: 5)"
    )

    args = parser.parse_args()

    copier = RealTraceCopier(args.source_dir, args.output_dir)
    copier.num_batches_to_copy = args.num_batches
    copier.copy_traces()


if __name__ == "__main__":
    main()
